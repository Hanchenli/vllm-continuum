# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the continuum scheduling policy's supporting state."""

import pytest

from vllm.v1.core.sched.continuum import (
    ContinuumJobRegistry,
    ContinuumRecorder,
    PinnedRequest,
    PinRegistry,
    ToolCallEstimator,
    get_tool_call_parser,
    job_key,
)
from vllm.v1.core.sched.request_queue import (
    ContinuumRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


class FakeRequest:
    """Minimal stand-in exposing only what the continuum state reads."""

    def __init__(
        self,
        request_id: str,
        job_id: str | None = None,
        arrival_time: float = 0.0,
        this_func_call: str | None = None,
        is_last_step: bool | None = None,
    ) -> None:
        self.request_id = request_id
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.this_func_call = this_func_call
        self.last_func_call: str | None = None
        self.is_last_step = is_last_step


########################################################################
# Tool call parsing
########################################################################


HERMES_OUTPUT = (
    'I will list the directory.\n<tool_call>\n{"name": "bash", '
    '"arguments": {"command": "ls -la"}}\n</tool_call>'
)
BASH_FENCE_OUTPUT = "Let me look around.\n```bash\nls -la\n```"


def test_bash_fence_parser_reads_fenced_command():
    parser = get_tool_call_parser("bash_fence")
    assert parser.parse(BASH_FENCE_OUTPUT) == "ls"


def test_bash_fence_parser_requires_exactly_one_fence():
    parser = get_tool_call_parser("bash_fence")
    assert parser.parse("```bash\nls\n```\n```bash\npwd\n```") is None
    assert parser.parse("no fence here") is None


def test_bash_fence_parser_does_not_understand_structured_tool_calls():
    """The pre-existing hardcoded parser cannot read structured tool calls.

    This is the failure mode the pluggable parser exists to fix: an agent
    emitting OpenAI/Hermes-shaped tool calls yields no tool name, so no
    request is ever pinned and the policy silently degrades to job-level
    FCFS. Pairs with the test below.
    """
    assert get_tool_call_parser("bash_fence").parse(HERMES_OUTPUT) is None


def test_hermes_parser_reads_structured_tool_calls():
    """Same input as above, now parsed because the parser is selectable."""
    assert get_tool_call_parser("hermes").parse(HERMES_OUTPUT) == "bash"


def test_json_parser_reads_bare_tool_call_object():
    parser = get_tool_call_parser("json")
    assert parser.parse('{"name": "read_file", "arguments": {}}') == "read_file"


def test_noop_parser_never_infers():
    assert get_tool_call_parser("none").parse(HERMES_OUTPUT) is None


def test_unknown_parser_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown continuum tool call parser"):
        get_tool_call_parser("does-not-exist")


########################################################################
# Estimator
########################################################################


def test_client_supplied_tool_call_is_not_overwritten_by_parsing():
    """A client that declares its tool call is authoritative.

    The previous implementation re-parsed server-side and overwrote the
    client's value, so a client whose output shape the parser did not
    recognise lost the metadata it had correctly supplied.
    """
    estimator = ToolCallEstimator(parser=get_tool_call_parser("bash_fence"))
    request = FakeRequest("r1", job_id="j1", this_func_call="search_files")
    estimator.request_finished(request, BASH_FENCE_OUTPUT)
    assert request.this_func_call == "search_files"


def test_parsing_fills_in_a_missing_tool_call():
    estimator = ToolCallEstimator(parser=get_tool_call_parser("hermes"))
    request = FakeRequest("r1", job_id="j1")
    estimator.request_finished(request, HERMES_OUTPUT)
    assert request.this_func_call == "bash"


def test_structured_tool_call_workload_is_pinnable():
    """A structured-tool-call turn must produce a non-zero pin duration.

    Guards the end-to-end intent: if this returns 0 the scheduler pins
    nothing and the arm measures plain FCFS instead of continuum.
    """
    estimator = ToolCallEstimator(parser=get_tool_call_parser("hermes"))
    request = FakeRequest("r1", job_id="j1")
    estimator.request_finished(request, HERMES_OUTPUT)
    assert estimator.pin_duration_for(request) > 0


def test_last_step_has_no_tool_call_to_pin_for():
    estimator = ToolCallEstimator(parser=get_tool_call_parser("hermes"))
    request = FakeRequest("r1", job_id="j1")
    estimator.request_finished(request, "just a final answer, no tool call")
    assert request.this_func_call is None
    assert estimator.pin_duration_for(request) == 0


def test_slow_tools_are_not_pinned():
    """A tool slower than the pin duration would expire its pin unused."""
    estimator = ToolCallEstimator(
        parser=get_tool_call_parser("hermes"), pin_duration_s=0.01
    )
    request = FakeRequest("r1", job_id="j1", this_func_call="slow_tool")
    estimator.request_finished(request, None)
    # Second turn arrives well after the pin duration.
    import time as _time

    _time.sleep(0.05)
    next_turn = FakeRequest("r2", job_id="j1", this_func_call="slow_tool")
    estimator.request_arrived(next_turn)
    assert estimator.get_exec_time("slow_tool") > 0.01
    assert estimator.pin_duration_for(next_turn) == 0


def test_estimator_backfills_last_func_call():
    estimator = ToolCallEstimator(parser=get_tool_call_parser("hermes"))
    first = FakeRequest("r1", job_id="j1", this_func_call="bash")
    estimator.request_finished(first, None)
    second = FakeRequest("r2", job_id="j1")
    estimator.request_arrived(second)
    assert second.last_func_call == "bash"


########################################################################
# Job registry
########################################################################


def test_requests_without_job_id_are_separate_jobs():
    """Anonymous requests must not collapse into one shared job.

    Sharing a single implicit job would make unrelated trajectories evict
    and unpin each other's KV.
    """
    a = FakeRequest("r1", job_id=None)
    b = FakeRequest("r2", job_id=None)
    assert job_key(a) != job_key(b)


def test_job_ordering_uses_first_turn_arrival():
    registry = ContinuumJobRegistry()
    first_turn = FakeRequest("r1", job_id="j1", arrival_time=10.0)
    registry.request_arrived(first_turn)
    # The scheduler records the pin while the turn is still live, then marks
    # it departed; that order is what keeps the job's arrival key alive.
    registry.set_pinned(first_turn, True)
    registry.request_departed(first_turn)

    later_turn = FakeRequest("r2", job_id="j1", arrival_time=99.0)
    registry.request_arrived(later_turn)
    # The job keeps its original position rather than going to the back.
    assert registry.first_arrival_time(later_turn) == 10.0


def test_set_pinned_recreates_a_reaped_job():
    """Pinning after the job was reaped must still be visible.

    Recording the pin only when the job state happens to exist made a
    mis-ordered caller silently lose every pin.
    """
    registry = ContinuumJobRegistry()
    request = FakeRequest("r1", job_id="j1", arrival_time=3.0)
    registry.request_arrived(request)
    registry.request_departed(request)
    assert len(registry) == 0
    registry.set_pinned(request, True)
    assert registry.is_pinned("j1")


def test_unknown_job_falls_back_to_request_arrival_time():
    registry = ContinuumJobRegistry()
    request = FakeRequest("r1", job_id="j1", arrival_time=7.0)
    assert registry.first_arrival_time(request) == 7.0


def test_pinned_job_state_survives_departure():
    registry = ContinuumJobRegistry()
    request = FakeRequest("r1", job_id="j1", arrival_time=1.0)
    registry.request_arrived(request)
    registry.set_pinned(request, True)
    registry.request_departed(request)
    assert registry.is_pinned("j1")
    registry.set_pinned(request, False)
    assert not registry.is_pinned("j1")
    assert len(registry) == 0


########################################################################
# Pin registry
########################################################################


def _pin(request, expires_at: float, num_blocks: int = 1) -> PinnedRequest:
    return PinnedRequest(
        request=request, expires_at=expires_at, blocks=[object()] * num_blocks
    )


def test_pin_budget_is_enforced():
    registry = PinRegistry(max_pinned_blocks=4)
    registry.add(_pin(FakeRequest("r1", job_id="j1"), 1.0, num_blocks=3))
    assert registry.num_pinned_blocks == 3
    assert registry.has_capacity_for(1)
    assert not registry.has_capacity_for(2)


def test_unlimited_pin_budget_by_default():
    registry = PinRegistry()
    assert registry.has_capacity_for(10**9)


def test_pop_expired_does_not_skip_neighbours():
    """Every expired entry must be returned in a single pass.

    A previous implementation removed entries while iterating the same list,
    which silently skipped the element after each removal and leaked pins.
    """
    registry = PinRegistry()
    for i in range(6):
        registry.add(_pin(FakeRequest(f"r{i}", job_id=f"j{i}"), expires_at=1.0))
    expired = registry.pop_expired(now=2.0, exclude_jobs=set())
    assert len(expired) == 6
    assert len(registry) == 0
    assert registry.num_pinned_blocks == 0


def test_pop_expired_keeps_jobs_with_a_queued_turn():
    registry = PinRegistry()
    registry.add(_pin(FakeRequest("r1", job_id="j1"), expires_at=1.0))
    registry.add(_pin(FakeRequest("r2", job_id="j2"), expires_at=1.0))
    expired = registry.pop_expired(now=2.0, exclude_jobs={"j2"})
    assert [job_key(e.request) for e in expired] == ["j1"]
    assert len(registry) == 1


def test_pop_expired_keeps_unexpired_entries():
    registry = PinRegistry()
    registry.add(_pin(FakeRequest("r1", job_id="j1"), expires_at=5.0))
    assert registry.pop_expired(now=2.0, exclude_jobs=set()) == []
    assert len(registry) == 1


def test_pop_for_job_releases_every_entry():
    registry = PinRegistry()
    registry.add(_pin(FakeRequest("r1", job_id="j1"), expires_at=1.0, num_blocks=2))
    registry.add(_pin(FakeRequest("r2", job_id="j1"), expires_at=1.0, num_blocks=3))
    registry.add(_pin(FakeRequest("r3", job_id="j2"), expires_at=1.0, num_blocks=4))
    popped = registry.pop_for_job("j1")
    assert len(popped) == 2
    assert registry.num_pinned_blocks == 4


def test_pop_latest_expiring_is_the_reclaim_order():
    registry = PinRegistry()
    registry.add(_pin(FakeRequest("soon", job_id="j1"), expires_at=1.0))
    registry.add(_pin(FakeRequest("later", job_id="j2"), expires_at=9.0))
    entry = registry.pop_latest_expiring()
    assert entry is not None
    assert entry.request.request_id == "later"


def test_pop_latest_expiring_on_empty_registry():
    assert PinRegistry().pop_latest_expiring() is None


########################################################################
# Request queue
########################################################################


def _queue(registry: ContinuumJobRegistry) -> ContinuumRequestQueue:
    queue = create_request_queue(SchedulingPolicy.CONTINUUM, registry)
    assert isinstance(queue, ContinuumRequestQueue)
    return queue


def test_continuum_queue_requires_a_registry():
    with pytest.raises(ValueError, match="requires a job registry"):
        create_request_queue(SchedulingPolicy.CONTINUUM)


def test_queue_orders_by_job_arrival_not_request_arrival():
    registry = ContinuumJobRegistry()
    old_job_turn = FakeRequest("r2", job_id="j1", arrival_time=50.0)
    new_job_turn = FakeRequest("r1", job_id="j2", arrival_time=20.0)
    # j1's first turn arrived before j2 existed.
    registry.request_arrived(FakeRequest("r0", job_id="j1", arrival_time=10.0))
    registry.request_arrived(new_job_turn)

    queue = _queue(registry)
    queue.add_request(new_job_turn)
    queue.add_request(old_job_turn)
    assert queue.peek_request() is old_job_turn


def test_queue_prefers_pinned_jobs():
    registry = ContinuumJobRegistry()
    early = FakeRequest("r1", job_id="j1", arrival_time=1.0)
    late = FakeRequest("r2", job_id="j2", arrival_time=2.0)
    registry.request_arrived(early)
    registry.request_arrived(late)
    registry.set_pinned(late, True)

    queue = _queue(registry)
    queue.add_request(early)
    queue.add_request(late)
    # j2 is later but holds a pin that this turn will consume.
    assert queue.peek_request() is late


def test_queue_pop_removes_the_peeked_request():
    registry = ContinuumJobRegistry()
    a = FakeRequest("r1", job_id="j1", arrival_time=1.0)
    b = FakeRequest("r2", job_id="j2", arrival_time=2.0)
    for r in (a, b):
        registry.request_arrived(r)
    queue = _queue(registry)
    queue.add_request(b)
    queue.add_request(a)
    assert queue.pop_request() is a
    assert len(queue) == 1
    assert queue.pop_request() is b
    assert not queue


def test_queue_peek_on_empty_raises():
    with pytest.raises(IndexError):
        _queue(ContinuumJobRegistry()).peek_request()


def test_queue_remove_requests():
    registry = ContinuumJobRegistry()
    reqs = [FakeRequest(f"r{i}", job_id=f"j{i}", arrival_time=i) for i in range(4)]
    for r in reqs:
        registry.request_arrived(r)
    queue = _queue(registry)
    for r in reqs:
        queue.add_request(r)
    queue.remove_requests([reqs[1], reqs[3]])
    assert list(queue) == [reqs[0], reqs[2]]


########################################################################
# Recorder
########################################################################


def test_recorder_is_inert_without_an_output_dir():
    recorder = ContinuumRecorder(None)
    assert not recorder.enabled
    recorder.request_arrived(FakeRequest("r1", job_id="j1"))
    recorder.write()
    assert recorder.job_history == {}


def test_recorder_tolerates_a_finish_without_an_arrival(tmp_path):
    recorder = ContinuumRecorder(str(tmp_path))
    # No matching request_arrived: must not raise.
    recorder.request_finished(FakeRequest("r1", job_id="j1"))
    assert "j1" in recorder.job_history


def test_recorder_writes_history(tmp_path):
    import json

    recorder = ContinuumRecorder(str(tmp_path))
    request = FakeRequest("r1", job_id="j1")
    recorder.request_arrived(request)
    recorder.request_pinned(request, num_blocks=3)
    recorder.write()
    written = json.loads((tmp_path / "scheduler_timestamps").read_text())
    assert [event for event in written["j1"] if "pinned_time" in event]
