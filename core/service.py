"""对比服务：解码 → 对齐 → 依次调用七种方法 → 汇总 + 切面计时。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .aspect import AspectStore, TimingAspect, current_trace_id, new_trace_id, now_iso
from .imaging import MAX_SIDE, align_pair, decode_data_url
from .methods import (
    GrayFingerprintMethod,
    HsvRedMethod,
    OcrMethod,
    OrbAlignMethod,
    PixelDiffMethod,
    SsimMethod,
    YoloMethod,
)
from .methods.base import ChangeMethod
from .models import CompareResponse, MethodResult, build_consensus

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "aspect.jsonl"
VERSION = "1.0.0"


class CompareService:
    def __init__(self, weights: str | None = None) -> None:
        self.store = AspectStore(LOG_PATH)
        self.method_aspect = TimingAspect(self.store, pointcut=r"\.detect$")
        self.service_aspect = TimingAspect(self.store, pointcut=r"CompareService\.compare$")
        self._lock = threading.Lock()

        self.methods: list[ChangeMethod] = [
            PixelDiffMethod(),
            HsvRedMethod(),
            SsimMethod(),
            OcrMethod(),
            YoloMethod(weights),
            GrayFingerprintMethod(),
            OrbAlignMethod(),
        ]
        # AOP：按 pointcut 扫描每个实现类的 detect，换上计时代理
        for method in self.methods:
            self.method_aspect.weave(method)

        # 服务自身的 compare 也套一层，totalMs 才能覆盖“进入→返回”
        self.compare = self.service_aspect.wrap(self.compare, "CompareService.compare")

    # ---------------------------------------------------------------- 元信息
    def meta(self) -> dict[str, Any]:
        methods = []
        for m in sorted(self.methods, key=lambda x: x.order):
            item = m.meta()
            status = None
            if hasattr(m, "status"):
                try:
                    status = m.status()
                except Exception:
                    status = None
            item["status"] = status
            available = bool(getattr(m, "available", True))
            if isinstance(status, dict) and "available" in status:
                available = bool(status["available"])
            item["available"] = available
            item["unavailableReason"] = (
                (status or {}).get("reason") if not available else None
            )
            methods.append(item)
        return {
            "version": VERSION,
            "limits": {
                "maxSide": MAX_SIDE,
                "maxImageMb": 12,
                "maxBodyMb": 40,
            },
            "methods": methods,
            "aspect": {
                "enabled": True,
                "pointcut": "execution(*.detect(..))",
                "service": "CompareService.compare",
                "store": str(LOG_PATH),
            },
        }

    def warmup_async(self) -> None:
        """后台预热 OCR / YOLO，尽量让第一次点击不被冷启动拖慢。"""

        def run() -> None:
            for m in self.methods:
                fn = getattr(m, "warmup", None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass

        threading.Thread(target=run, name="model-warmup", daemon=True).start()

    # ---------------------------------------------------------------- 主流程
    def compare(self, payload: dict[str, Any]) -> CompareResponse:
        trace_id = current_trace_id() or new_trace_id()
        started_at = now_iso()
        before_raw = decode_data_url(payload.get("before"), "前图")
        after_raw = decode_data_url(payload.get("after"), "后图")
        before, after, image_info = align_pair(before_raw, after_raw)

        requested = payload.get("methods")
        if requested is None:
            # 默认只跑“当前可用”的方法：BYOM 未配置 YOLO 时自动跳过
            selected = [m for m in self.methods if getattr(m, "available", True)]
        else:
            if not isinstance(requested, list):
                raise ValueError("methods 必须是数组")
            wanted = {str(x) for x in requested}
            selected = [m for m in self.methods if m.key in wanted]
            if not selected:
                raise ValueError("没有选中任何有效方法")

        results: list[MethodResult] = []
        with self._lock:
            for m in selected:
                try:
                    result = m.detect(before, after)
                    results.append(result)
                except Exception as exc:
                    elapsed = 0.0
                    target = f"{type(m).__name__}.detect"
                    for rec in reversed(self.store.by_trace(trace_id)):
                        if rec.get("target") == target:
                            elapsed = float(rec.get("elapsedMs") or 0)
                            break
                    results.append(
                        MethodResult(
                            key=m.key,
                            name=m.name,
                            ok=False,
                            error=f"{type(exc).__name__}: {exc}",
                            elapsed_ms=elapsed,
                            summary=f"该方法执行失败：{type(exc).__name__}",
                        )
                    )

        consensus = build_consensus(results)
        records = self.store.by_trace(trace_id)
        timing = {
            "totalMs": 0.0,
            "serviceMs": 0.0,
            "methodMs": {r.key: round(r.elapsed_ms, 2) for r in results},
            "aspect": {
                "enabled": True,
                "pointcut": "execution(*.detect(..))",
                "traceId": trace_id,
                "records": records,
            },
        }

        return CompareResponse(
            request_id=trace_id,
            started_at=started_at,
            images=image_info,
            methods=results,
            consensus=consensus,
            timing=timing,
        )
