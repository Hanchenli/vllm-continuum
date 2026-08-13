# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Supporting state for the ``continuum`` scheduling policy.

The ``continuum`` policy treats a multi-turn agent trajectory as a single
*job*: every LLM call the agent makes carries the same ``job_id``. Between
turns the agent runs a tool, during which the request has finished and its KV
would normally be returned to the block pool. Continuum instead *pins* that KV
for a short TTL so the next turn of the same job hits it, and schedules
requests job-first-come-first-served with pinned jobs preferred.

Pinning composes with automatic prefix caching rather than replacing it: a
pinned request's blocks keep their references and stay in the prefix cache, so
they remain hit-able and simply cannot be evicted while the pin is held.

Request metadata is carried in ``SamplingParams.extra_args`` (populated from
the OpenAI-compatible ``vllm_xargs`` request field):

* ``job_id`` - trajectory identifier shared by every turn.
* ``is_last_step`` - the final turn of the trajectory; its KV is not pinned.
* ``this_func_call`` - the tool this turn will invoke. Optional; when absent
  the server may infer it from the generated text (see ``ToolCallParser``).
* ``last_func_call`` - the tool invoked by the previous turn. Optional;
  inferred server-side when absent.
"""

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Fallback pin duration, in seconds, used when no per-tool execution-time
# estimate is available yet.
DEFAULT_PIN_DURATION_S = 2.0


def job_key(request: "Request") -> str:
    """The job a request belongs to.

    Requests that carry no ``job_id`` are treated as single-turn jobs of their
    own rather than being merged into one shared anonymous job.
    """
    return request.job_id if request.job_id is not None else request.request_id


########################################################################
# Tool-call parsing
########################################################################


class ToolCallParser(ABC):
    """Extracts the name of the tool a completion is about to invoke.

    Continuum uses the tool name only as a key for its per-tool
    execution-time estimate, so parsers may return any stable identifier.
    """

    @abstractmethod
    def parse(self, text: str) -> str | None:
        """Return the tool name found in ``text``, or None."""
        raise NotImplementedError


class BashFenceToolCallParser(ToolCallParser):
    """Agents that act by emitting a single ```bash fenced command.

    This is the shape used by mini-swe-agent and is kept as the default for
    backwards compatibility.
    """

    _PATTERN = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

    def parse(self, text: str) -> str | None:
        actions = self._PATTERN.findall(text)
        if len(actions) != 1:
            return None
        words = actions[0].strip().split()
        return words[0] if words else None


class HermesToolCallParser(ToolCallParser):
    """Agents emitting Hermes-style ``<tool_call>{"name": ...}</tool_call>``."""

    _PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    _NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')

    def parse(self, text: str) -> str | None:
        blocks = self._PATTERN.findall(text)
        if not blocks:
            return None
        # Use the first tool call; a turn that fans out to several tools is
        # keyed by the first one, which is enough for a duration estimate.
        match = self._NAME_PATTERN.search(blocks[0])
        return match.group(1) if match else None


class JSONToolCallParser(ToolCallParser):
    """Agents emitting a bare JSON object with a ``name`` field."""

    _NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')

    def parse(self, text: str) -> str | None:
        match = self._NAME_PATTERN.search(text)
        return match.group(1) if match else None


class NoOpToolCallParser(ToolCallParser):
    """Never infers a tool name; use when the client always supplies one."""

    def parse(self, text: str) -> str | None:
        return None


TOOL_CALL_PARSERS: dict[str, type[ToolCallParser]] = {
    "bash_fence": BashFenceToolCallParser,
    "hermes": HermesToolCallParser,
    "json": JSONToolCallParser,
    "none": NoOpToolCallParser,
}


def get_tool_call_parser(name: str) -> ToolCallParser:
    try:
        return TOOL_CALL_PARSERS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown continuum tool call parser {name!r}. "
            f"Available: {sorted(TOOL_CALL_PARSERS)}."
        ) from None


########################################################################
# Job bookkeeping
########################################################################


@dataclass
class _JobState:
    first_arrival_time: float
    live_requests: int = 0
    pinned: bool = False


class ContinuumJobRegistry:
    """Tracks per-job arrival order, liveness, and pin status.

    A single registry is shared by the scheduler and its waiting queues so
    that both observe the same job ordering; keeping two copies (as an
    earlier revision did) let them disagree about a job's arrival time.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, _JobState] = {}

    def request_arrived(self, request: "Request") -> None:
        key = job_key(request)
        state = self._jobs.get(key)
        if state is None:
            self._jobs[key] = _JobState(first_arrival_time=request.arrival_time)
            state = self._jobs[key]
        state.live_requests += 1

    def request_departed(self, request: "Request") -> None:
        key = job_key(request)
        state = self._jobs.get(key)
        if state is None:
            return
        state.live_requests -= 1
        # Retain the entry while the job still holds a pin so that the next
        # turn keeps its original position in the job ordering.
        if state.live_requests <= 0 and not state.pinned:
            del self._jobs[key]

    def set_pinned(self, request: "Request", pinned: bool) -> None:
        key = job_key(request)
        state = self._jobs.get(key)
        if state is None:
            if not pinned:
                return
            # The job was reaped before its pin was recorded. Recreate it so
            # the pin stays visible to the queue; the recovered ordering key
            # is this turn's arrival rather than the job's first.
            state = _JobState(first_arrival_time=request.arrival_time)
            self._jobs[key] = state
        state.pinned = pinned
        if not pinned and state.live_requests <= 0:
            del self._jobs[key]

    def is_pinned(self, job_id: str) -> bool:
        state = self._jobs.get(job_id)
        return state is not None and state.pinned

    def first_arrival_time(self, request: "Request") -> float:
        """Arrival time of the job's first turn, for job-level FCFS.

        Falls back to the request's own arrival time for jobs the registry
        has not seen, so ordering never depends on a missing entry.
        """
        state = self._jobs.get(job_key(request))
        return state.first_arrival_time if state is not None else request.arrival_time

    def __len__(self) -> int:
        return len(self._jobs)


########################################################################
# Pin bookkeeping
########################################################################


@dataclass
class PinnedRequest:
    """A finished request whose KV blocks are being held for its next turn."""

    request: "Request"
    expires_at: float
    blocks: list["KVCacheBlock"]

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


@dataclass
class PinRegistry:
    """Owns pinned KV and enforces the pinned-block budget.

    ``max_pinned_blocks`` of ``None`` means unlimited, which reproduces the
    original behaviour; setting it bounds how much of the block pool pinning
    may hold so that admission degrades gracefully instead of starving.
    """

    max_pinned_blocks: int | None = None
    _entries: list[PinnedRequest] = field(default_factory=list)
    _num_pinned_blocks: int = 0

    @property
    def num_pinned_blocks(self) -> int:
        return self._num_pinned_blocks

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __iter__(self) -> Iterator[PinnedRequest]:
        return iter(self._entries)

    def has_capacity_for(self, num_blocks: int) -> bool:
        if self.max_pinned_blocks is None:
            return True
        return self._num_pinned_blocks + num_blocks <= self.max_pinned_blocks

    def add(self, entry: PinnedRequest) -> None:
        self._entries.append(entry)
        self._num_pinned_blocks += entry.num_blocks

    def remove(self, entry: PinnedRequest) -> None:
        self._entries.remove(entry)
        self._num_pinned_blocks -= entry.num_blocks

    def job_ids(self) -> set[str]:
        return {job_key(entry.request) for entry in self._entries}

    def pop_expired(self, now: float, exclude_jobs: set[str]) -> list[PinnedRequest]:
        """Remove and return entries whose TTL has elapsed.

        Jobs in ``exclude_jobs`` (i.e. whose next turn is already queued) keep
        their pin: the turn about to be scheduled is exactly the one the pin
        exists for.

        The expired entries are collected before any mutation so that removing
        them cannot cause the scan to skip a neighbour.
        """
        expired = [
            entry
            for entry in self._entries
            if now >= entry.expires_at and job_key(entry.request) not in exclude_jobs
        ]
        for entry in expired:
            self.remove(entry)
        return expired

    def pop_for_job(self, job_id: str) -> list[PinnedRequest]:
        """Remove and return every entry belonging to ``job_id``."""
        matching = [
            entry for entry in self._entries if job_key(entry.request) == job_id
        ]
        for entry in matching:
            self.remove(entry)
        return matching

    def pop_latest_expiring(self) -> PinnedRequest | None:
        """Remove and return the entry that would expire last.

        Used as a last-resort release when no running request can be
        preempted. Releasing the latest-expiring pin first keeps the pins that
        are closest to being claimed by their next turn.
        """
        if not self._entries:
            return None
        entry = max(self._entries, key=lambda e: e.expires_at)
        self.remove(entry)
        return entry


########################################################################
# Tool execution-time estimation
########################################################################


class ToolCallEstimator:
    """Estimates how long a job will stall in its tool call.

    The estimate keys off the tool name and is a running mean of observed
    gaps between one turn finishing and the next turn arriving. A job whose
    tool is slower than the pin duration is not worth pinning: its KV would
    expire before the next turn arrives.

    NOTE: this is the fixed-threshold policy from the Continuum preview code,
    not the estimator described in the paper.
    """

    def __init__(
        self,
        parser: ToolCallParser | None = None,
        pin_duration_s: float = DEFAULT_PIN_DURATION_S,
    ) -> None:
        self.parser = parser if parser is not None else BashFenceToolCallParser()
        self.pin_duration_s = pin_duration_s
        self._mean_exec_time: dict[str, float] = {}
        self._samples: dict[str, list[float]] = {}
        # job_id -> (departure_time, func_call) of the most recent finished turn
        self._last_departure: dict[str, tuple[float, str | None]] = {}

    def get_exec_time(self, func: str) -> float | None:
        return self._mean_exec_time.get(func)

    def request_arrived(self, request: "Request") -> None:
        """Record a turn's arrival, updating the previous turn's tool timing."""
        key = job_key(request)
        previous = self._last_departure.pop(key, None)
        if previous is None:
            return
        departure_time, func = previous
        if request.last_func_call is None:
            request.last_func_call = func
        if func is None:
            return
        observed = time.monotonic() - departure_time
        samples = self._samples.setdefault(func, [])
        samples.append(observed)
        self._mean_exec_time[func] = sum(samples) / len(samples)

    def request_finished(self, request: "Request", output_text: str | None) -> None:
        """Record a turn's completion and the tool it is about to invoke.

        A client-supplied ``this_func_call`` is authoritative and is never
        overwritten by server-side parsing; parsing only fills the gap when
        the client did not say.
        """
        if request.this_func_call is None and output_text is not None:
            try:
                request.this_func_call = self.parser.parse(output_text)
            except Exception:
                logger.exception(
                    "Continuum tool call parser failed for request %s",
                    request.request_id,
                )
        self._last_departure[job_key(request)] = (
            time.monotonic(),
            request.this_func_call,
        )

    def forget_job(self, job_id: str) -> None:
        self._last_departure.pop(job_id, None)

    def pin_duration_for(self, request: "Request") -> float:
        """How long to pin this request's KV; 0 means do not pin."""
        if request.this_func_call is None:
            return 0.0
        exec_time = self.get_exec_time(request.this_func_call)
        if exec_time is not None and exec_time > self.pin_duration_s:
            # The next turn will not arrive before the pin expires.
            return 0.0
        return self.pin_duration_s


########################################################################
# Experiment instrumentation
########################################################################


class ContinuumRecorder:
    """Per-job event log, written out at shutdown for offline analysis.

    Enabled by setting ``RUN_OUTPUT_DIR``; a no-op otherwise so that the
    policy carries no file-writing side effect by default.
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir
        self.job_history: dict[str, list[dict[str, Any]]] = {}

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def _record(self, request: "Request", event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        # setdefault: a request may finish without a recorded arrival if the
        # recorder was attached mid-flight.
        self.job_history.setdefault(job_key(request), []).append(event)

    def request_arrived(self, request: "Request") -> None:
        self._record(request, {"request_arrival_time": time.time()})

    def request_finished(self, request: "Request") -> None:
        self._record(request, {"request_departure_time": time.time()})

    def request_preempted(self, request: "Request") -> None:
        self._record(request, {"request_preempted_time": time.time()})

    def request_pinned(self, request: "Request", num_blocks: int) -> None:
        self._record(request, {"pinned_time": time.time(), "num_blocks": num_blocks})

    def request_unpinned(self, request: "Request") -> None:
        self._record(request, {"unpinned_time": time.time()})

    def request_scheduled(
        self, request: "Request", prompt_length: int, hit_length: int, resumed: bool
    ) -> None:
        self._record(
            request,
            {
                "evicted_to_running" if resumed else "waiting_to_running": time.time(),
                "prompt_length": prompt_length,
                "hit_length": hit_length,
            },
        )

    def write(self) -> None:
        if not self.enabled:
            return
        import json
        import os

        assert self.output_dir is not None
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            final_path = os.path.join(self.output_dir, "scheduler_timestamps")
            tmp_path = final_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self.job_history, f, indent=2)
            os.replace(tmp_path, final_path)
        except OSError:
            logger.exception("Failed to write continuum scheduler timestamps")
