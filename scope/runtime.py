"""Small module scheduler and telemetry. Source clocks never mix with host clocks."""

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import StrEnum
import logging
import math
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class Health(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class ModuleConfig:
    enabled: bool = True
    max_hz: float | None = None
    every_n: int = 1
    stale_after_s: float = .5

    def __post_init__(self):
        if (self.max_hz is not None and (not math.isfinite(self.max_hz) or self.max_hz <= 0)
                or self.every_n < 1 or not math.isfinite(self.stale_after_s)
                or self.stale_after_s <= 0):
            raise ValueError("Positive finite rates, stride and stale interval required")


@dataclass
class ModuleState:
    config: ModuleConfig
    status: Health = Health.UNAVAILABLE
    reason: str = "No successful update"
    source: str | None = None
    last_output: object = None
    last_success_s: float | None = None
    last_attempt_s: float | None = None
    input_timestamp_s: float | None = None
    host_receive_monotonic_s: float | None = None
    host_complete_monotonic_s: float | None = None
    calls: int = 0
    successes: int = 0
    failures: int = 0
    restarts: int = 0
    dropped_inputs: int = 0
    queue_depth: int = 0
    durations_ms: deque = field(default_factory=lambda: deque(maxlen=10000))
    cpu_ms: deque = field(default_factory=lambda: deque(maxlen=10000))
    success_times_s: deque = field(default_factory=lambda: deque(maxlen=10000))
    success_host_times_s: deque = field(default_factory=lambda: deque(maxlen=10000))
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=8))


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "mean_ms": float(a.mean()) if len(a) else None,
            "median_ms": float(np.median(a)) if len(a) else None,
            "p95_ms": float(np.percentile(a, 95)) if len(a) else None}


class Runtime:
    """Mutable configuration is the future GUI boundary. Each run is a failure boundary.

    A returned None means no *new* output; retained output is only exposed with age
    through state/snapshot. Consumers must explicitly choose whether to reuse it.
    """
    def __init__(self, configs=None):
        self.modules = {name: ModuleState(config) for name, config in (configs or {}).items()}

    def configure(self, name, config):
        state = self.modules.setdefault(name, ModuleState(config))
        if not state.config.enabled and config.enabled:
            state.restarts += 1
            state.last_attempt_s = None
            state.status, state.reason = Health.UNAVAILABLE, "Re-enabled; awaiting fresh input"
        state.config = config
        if not config.enabled:
            state.status, state.reason = Health.DISABLED, "Disabled by runtime configuration"

    def state(self, name):
        return self.modules.setdefault(name, ModuleState(ModuleConfig()))

    def mark(self, name, status, reason):
        state = self.state(name)
        state.status, state.reason = status, reason

    def due(self, name, timestamp_s):
        if not math.isfinite(timestamp_s):
            raise ValueError("Source timestamp must be finite")
        state = self.state(name)
        state.calls += 1
        cfg = state.config
        if not cfg.enabled:
            self.mark(name, Health.DISABLED, "Disabled by runtime configuration")
            return False
        if ((state.calls - 1) % cfg.every_n or cfg.max_hz is not None
                and state.last_attempt_s is not None
                and timestamp_s - state.last_attempt_s < 1 / cfg.max_hz - 1e-8):
            state.dropped_inputs += 1
            return False
        state.last_attempt_s = timestamp_s
        return True

    def run(self, name, timestamp_s, function, *, input_s=None, source=None,
            host_receive_s=None, scheduled=True):
        if not self.state(name).config.enabled:
            self.mark(name,Health.DISABLED,"Disabled by runtime configuration")
            return None
        if scheduled and not self.due(name, timestamp_s):
            return None
        state = self.state(name)
        state.input_timestamp_s = timestamp_s if input_s is None else input_s
        state.host_receive_monotonic_s = host_receive_s
        state.source = source or state.source
        start, cpu = time.perf_counter(), time.thread_time()
        try:
            output = function()
        except Exception as exc:
            # This is a subsystem boundary, not a blanket catch around the app.
            # Programmer failures retain full traceback in the logger.
            log.exception("Module %s failed", name)
            state.failures += 1
            state.recent_failures.append({"timestamp_s": timestamp_s,
                                          "type": type(exc).__name__, "reason": str(exc)})
            self.mark(name, Health.FAILED, f"{type(exc).__name__}: {exc}")
            return None
        finally:
            state.durations_ms.append((time.perf_counter() - start) * 1000)
            state.cpu_ms.append((time.thread_time() - cpu) * 1000)
            state.host_complete_monotonic_s = time.monotonic()
        state.last_output = output
        state.last_success_s = timestamp_s
        state.success_times_s.append(timestamp_s)
        state.success_host_times_s.append(state.host_complete_monotonic_s)
        state.successes += 1
        self.mark(name, Health.OK, "Fresh output")
        return output

    def snapshot(self, now_s):
        result = {}
        for name, state in self.modules.items():
            age = None if state.last_success_s is None else max(0., now_s-state.last_success_s)
            status, reason = state.status, state.reason
            if not state.config.enabled:
                status, reason = Health.DISABLED, "Disabled by runtime configuration"
            elif age is not None and age > state.config.stale_after_s and status not in (
                    Health.FAILED, Health.UNAVAILABLE):
                status, reason = Health.STALE, "Last valid output exceeded its age limit"
            times = state.success_times_s
            rate = (len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None
            host_times = state.success_host_times_s
            host_rate = (len(host_times)-1)/(host_times[-1]-host_times[0]) if len(host_times)>1 and host_times[-1]>host_times[0] else None
            result[name] = {
                "status": str(status), "reason": reason, "source": state.source,
                "config": asdict(state.config), "last_success_source_s": state.last_success_s,
                "data_age_s": age,
                "input_age_s": None if state.input_timestamp_s is None else max(0., now_s-state.input_timestamp_s),
                "host_receive_monotonic_s": state.host_receive_monotonic_s,
                "host_complete_monotonic_s": state.host_complete_monotonic_s,
                "host_end_to_end_ms": None if state.host_receive_monotonic_s is None or state.host_complete_monotonic_s is None else
                    (state.host_complete_monotonic_s-state.host_receive_monotonic_s)*1000,
                "source_update_hz": rate, "host_update_hz":host_rate, "latency": distribution(state.durations_ms),
                "cpu": distribution(state.cpu_ms), "gpu_inference_ms": None,
                "successes": state.successes, "failures": state.failures,
                "restarts": state.restarts, "dropped_inputs": state.dropped_inputs,
                "queue_depth": state.queue_depth, "recent_failures": list(state.recent_failures),
            }
        return result


class LatestWorker:
    """One running input + one replaceable pending input; no unbounded backlog.

    Dedicated daemon workers isolate slow inference from map/other camera ticks.
    Shutdown never waits indefinitely on a third-party model. Results carry their
    original input; they must pass age/dependency checks before consumption.
    """
    def __init__(self, function):
        self.function = function
        self.pending = None
        self.completed = deque(maxlen=1)
        self.dropped = 0
        self.closed = False
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, item):
        with self.condition:
            if self.closed:
                raise RuntimeError("Worker is closed")
            if self.pending is not None:
                self.dropped += 1
            self.pending = item
            self.condition.notify()

    def poll(self):
        with self.condition:
            return self.completed.popleft() if self.completed else None

    def _loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                item, self.pending = self.pending, None
            start = time.perf_counter()
            try:
                value, error = self.function(item), None
            except Exception as exc:
                log.exception("Inference worker failed")
                value, error = None, f"{type(exc).__name__}: {exc}"
            with self.condition:
                if self.completed:
                    self.dropped += 1
                self.completed.append((item, value, error, (time.perf_counter()-start)*1000))

    @property
    def queue_depth(self):
        with self.condition:
            return int(self.pending is not None)

    def close(self,timeout_s=.1):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify()
        self.thread.join(timeout=timeout_s)
        if self.thread.is_alive():
            log.warning("Inference worker still finishing after shutdown timeout")
        return not self.thread.is_alive()
