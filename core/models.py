"""统一的输出契约。

七种方法各说各话，前端只认这一份结构：

    MethodResult
      ├─ 结论      hasChange / changeType / confidence / summary
      ├─ 证据      regions[]（对齐后画布坐标）
      ├─ 展示      metrics[]（给人看的指标）/ details{}（给程序看的原始值）
      └─ 计时      elapsedMs（由 AOP 切面在方法返回时写入）

坐标口径：**对齐后画布的像素**。两张图尺寸不一致时，后图会等比缩放到前图尺寸，
`images.aligned=true`，前端按图片实际显示尺寸换算即可（框坐标不需要前端再缩放）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChangeType = Literal["none", "added", "removed", "moved", "content", "unknown"]

CHANGE_TYPE_LABELS: dict[str, str] = {
    "none": "无变化",
    "added": "新增",
    "removed": "消失",
    "moved": "位置变化",
    "content": "内容变化",
    "unknown": "无法判定",
}

# 区域语义：前端按 kind 上色
#   added   新增的内容
#   removed 消失的内容
#   moved   移位后的位置
#   changed 发生变化的区域（方法无法细分新增/消失）
#   object  目标检测命中但未变化的辅助框
REGION_KINDS = ("added", "removed", "moved", "changed", "object")


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    label: str = ""
    kind: str = "changed"
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "label": self.label,
            "kind": self.kind if self.kind in REGION_KINDS else "changed",
        }
        if self.score is not None:
            d["score"] = round(float(self.score), 4)
        return d


@dataclass
class Metric:
    key: str
    label: str
    value: Any
    unit: str = ""
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "hint": self.hint,
        }


@dataclass
class MethodResult:
    key: str
    name: str
    ok: bool = True
    error: str | None = None
    elapsed_ms: float = 0.0
    has_change: bool = False
    change_type: ChangeType = "none"
    confidence: float = 0.0
    summary: str = ""
    regions: list[Region] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    overlay: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "ok": self.ok,
            "error": self.error,
            "elapsedMs": round(float(self.elapsed_ms), 2),
            "hasChange": bool(self.has_change),
            "changeType": self.change_type,
            "changeTypeLabel": CHANGE_TYPE_LABELS.get(self.change_type, self.change_type),
            "confidence": round(float(self.confidence), 4),
            "summary": self.summary,
            "regions": [r.to_dict() for r in self.regions],
            "metrics": [m.to_dict() for m in self.metrics],
            "details": self.details,
            "overlay": self.overlay,
        }


@dataclass
class CompareResponse:
    request_id: str
    started_at: str
    images: dict[str, Any]
    methods: list[MethodResult]
    consensus: dict[str, Any]
    timing: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "requestId": self.request_id,
            "startedAt": self.started_at,
            "images": self.images,
            "methods": [m.to_dict() for m in self.methods],
            "consensus": self.consensus,
            "timing": self.timing,
        }


def build_consensus(results: list[MethodResult]) -> dict[str, Any]:
    """多数投票 —— 只做「方法之间是否一致」的汇总，不冒充准确率。"""
    ok = [r for r in results if r.ok]
    if not ok:
        return {
            "hasChange": None,
            "changeType": "unknown",
            "changeTypeLabel": CHANGE_TYPE_LABELS["unknown"],
            "agreement": 0.0,
            "votes": {"change": 0, "noChange": 0},
            "typeVotes": {},
            "summary": "所有方法都执行失败，无法给出综合判断。",
        }

    change_votes = sum(1 for r in ok if r.has_change)
    no_votes = len(ok) - change_votes
    has_change = change_votes > no_votes
    agreement = max(change_votes, no_votes) / len(ok)

    type_counts: dict[str, int] = {}
    for r in ok:
        t = r.change_type if r.change_type in CHANGE_TYPE_LABELS else "unknown"
        type_counts[t] = type_counts.get(t, 0) + 1

    chosen = "none"
    if has_change:
        specific = {t: n for t, n in type_counts.items() if t in ("added", "removed", "moved")}
        # 有方法能说出具体类型时优先采用，避免“内容变化”把“红点新增”盖掉
        if specific:
            chosen = max(specific.items(), key=lambda kv: (kv[1], kv[0] != "content"))[0]
        elif type_counts.get("content"):
            chosen = "content"
        else:
            chosen = "unknown"
    else:
        chosen = "none"

    if has_change:
        summary = (
            f"{change_votes}/{len(ok)} 个方法判定「发生变化」，"
            f"综合类型：{CHANGE_TYPE_LABELS[chosen]}。"
        )
        if len(set(type_counts)) > 1:
            summary += " 各方法的类型口径不同，建议结合下方证据交叉判断。"
    else:
        summary = f"{no_votes}/{len(ok)} 个方法判定「未变化」。"

    return {
        "hasChange": has_change,
        "changeType": chosen,
        "changeTypeLabel": CHANGE_TYPE_LABELS[chosen],
        "agreement": round(agreement, 4),
        "votes": {"change": change_votes, "noChange": no_votes},
        "typeVotes": type_counts,
        "summary": summary,
    }
