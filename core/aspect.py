"""AOP 计时切面（Python 版 @Around）。

Spring AOP 的核心概念在这里一一对应：

    @Aspect          →  TimingAspect
    @Pointcut        →  pointcut 正则（默认 execution(*.detect(..))）
    @Around          →  wrap()：方法进入时 perf_counter_ns()，退出（含异常）时结算
    JoinPoint        →  wrapper 里的 fn / args / kwargs
    ProceedingJoinPoint → 调用被包装的 fn

它不是装饰器语法糖那么简单：服务启动时用 **pointcut 扫描** 所有方法实现类的
`detect`，动态替换成计时代理。新增一个方法类，只要方法名匹配 `.detect`，
不用改切面就会自动被计时。
"""

from __future__ import annotations

import contextvars
import itertools
import json
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "image_diff_trace_id", default=None
)
_seq = itertools.count(1)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class AspectStore:
    """切面记录仓库：内存环形队列 + JSONL 落盘，便于页面和事后排查。"""

    def __init__(self, log_path: Path | None = None, maxlen: int = 400) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(rec)
            if self._log_path:
                try:
                    with self._log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except OSError:
                    pass

    def by_trace(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self._records if r.get("traceId") == trace_id]

    def recent(self, limit: int = 60) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)[-limit:][::-1]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class TimingAspect:
    """记录「函数被调用 → 函数返回」的墙钟耗时，异常也记。"""

    def __init__(self, store: AspectStore, pointcut: str = r"\.detect$") -> None:
        self.store = store
        self.pointcut = pointcut
        self._re = re.compile(pointcut)

    def matches(self, qualname: str) -> bool:
        return bool(self._re.search(qualname))

    def weave(self, obj: Any, method_name: str = "detect") -> Any:
        """按 pointcut 给对象的方法套上代理；返回 obj，方便链式调用。"""
        qualname = f"{type(obj).__name__}.{method_name}"
        if not self.matches(qualname):
            return obj
        fn = getattr(obj, method_name, None)
        if callable(fn):
            setattr(obj, method_name, self.wrap(fn, qualname))
        return obj

    def wrap(self, fn: Callable[..., Any], label: str | None = None) -> Callable[..., Any]:
        if getattr(fn, "__timing_aspect__", False):
            return fn

        aspect = self
        target = label or getattr(fn, "__qualname__", getattr(fn, "__name__", "callable"))

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            inherited = _trace_id.get()
            trace_id = inherited or new_trace_id()
            token = None
            if inherited is None:
                token = _trace_id.set(trace_id)

            started_at = now_iso()
            start_ns = time.perf_counter_ns()
            status = "ok"
            error: str | None = None
            result: Any = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as exc:  # 记录后原样抛出，切面不吞异常
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                rec = {
                    "seq": next(_seq),
                    "traceId": trace_id,
                    "target": target,
                    "startedAt": started_at,
                    "elapsedMs": round(elapsed_ms, 2),
                    "status": status,
                    "error": error,
                    "thread": threading.current_thread().name,
                }
                aspect.store.record(rec)

                # 返回值本身带计时字段时，把切面测到的时间写回去（业务代码不用自己算）
                if result is not None and hasattr(result, "elapsed_ms"):
                    try:
                        result.elapsed_ms = elapsed_ms
                    except Exception:
                        pass
                if result is not None and hasattr(result, "timing") and isinstance(
                    getattr(result, "timing", None), dict
                ):
                    try:
                        timing = result.timing
                        timing["serviceMs"] = round(elapsed_ms, 2)
                        timing["totalMs"] = round(elapsed_ms, 2)
                        aspect_block = timing.setdefault("aspect", {})
                        records = aspect_block.setdefault("records", [])
                        # 服务自身的记录在方法记录之后产生，补进响应
                        records.append(rec)
                    except Exception:
                        pass
                if token is not None:
                    _trace_id.reset(token)

        setattr(wrapper, "__timing_aspect__", True)
        return wrapper


def current_trace_id() -> str | None:
    return _trace_id.get()
