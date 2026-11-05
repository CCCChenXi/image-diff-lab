"""方法 5：YOLO 目标检测 + 目标内像素比对。

回答：**变化发生在哪个 UI 元素里？**
纯 YOLO 不认“红点”这个类（自定义模型只有会话行、气泡、输入框等），
所以这里用它先框出 UI 元素，再对同一元素的前后裁剪做像素差。
这样既能定位到“哪一个联系人”，又不会把整张图的变化混在一起。
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import load_dotenv
from ..imaging import render_overlay
from ..models import MethodResult, Metric, Region
from .base import ChangeMethod, clamp

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models"
WECHAT = ROOT.parent / "wechat-vision"
RUNS = WECHAT / "training" / "runs"
PREFERRED = RUNS / "yolo26n_960_v3" / "weights" / "best.pt"
ENV_KEY = "YOLO_WEIGHTS"


def discover_weight(explicit: str | None = None) -> tuple[Path | None, str]:
    """按优先级找权重，同时返回“找不到”的原因，给前端做友好提示。

    开源版是 BYOM：权重不随仓库提供。查找顺序：
      1. --weights 显式路径
      2. 环境变量 / .env 的 YOLO_WEIGHTS
      3. 项目内 models/*.pt
      4. 兼容本机旧的 wechat-vision 训练目录
    """
    load_dotenv()

    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file():
            return p, ""
        return None, f"指定的权重文件不存在：{explicit}"

    env_value = (os.environ.get(ENV_KEY) or "").strip()
    if env_value:
        p = Path(env_value)
        if not p.is_absolute():
            p = (ROOT / p)
        if p.is_file():
            return p, ""
        return None, f"环境变量 {ENV_KEY} 指向的文件不存在：{env_value}"

    if MODELS.is_dir():
        local = sorted(MODELS.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if local:
            return local[0], ""

    if PREFERRED.is_file():
        return PREFERRED, ""
    if RUNS.is_dir():
        bests = sorted(RUNS.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if bests:
            return bests[0], ""

    return None, (
        "未配置 YOLO 权重（BYOM）。请自行训练模型后把 best.pt 放到 models/，"
        f"或设置 {ENV_KEY} / 使用 --weights 指定路径。"
    )


@dataclass
class _Obj:
    cls_id: int
    cls_name: str
    score: float
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "cls": self.cls_name,
            "score": round(self.score, 4),
            "box": [self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0],
        }


def _iou(a: _Obj, b: _Obj) -> float:
    xx0, yy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    xx1, yy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0, xx1 - xx0), max(0, yy1 - yy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (a.x1 - a.x0) * (a.y1 - a.y0))
    area_b = max(1, (b.x1 - b.x0) * (b.y1 - b.y0))
    return inter / float(area_a + area_b - inter)


class YoloMethod(ChangeMethod):
    key = "yolo"
    name = "YOLO 目标检测"
    description = "自定义微信 UI 模型框出元素，再比较同一元素内的像素变化。"
    color = "#8338ec"
    cost = "GPU 几十毫秒 / CPU 数百毫秒"
    strength = "把变化定位到具体 UI 元素，适合复杂界面"
    blind_spot = "依赖模型能框到的类；红点本身不是独立类别"
    order = 5

    def __init__(self, weights: str | None = None) -> None:
        self.weight_path, self.unavailable_reason = discover_weight(weights)
        self.available = self.weight_path is not None
        if self.available and find_spec("ultralytics") is None:
            self.available = False
            self.unavailable_reason = (
                "已找到 YOLO 权重，但未安装 ultralytics。"
                "请安装 requirements-yolo.txt（AGPL-3.0）。"
            )
        self.imgsz = 960
        self.conf = 0.25
        self.device = "cpu"
        self._model: Any = None
        self._lock = threading.Lock()
        self._ready = False
        self._warm = False
        self._names: dict[int, str] = {}

    # ---------------------------------------------------------------- 模型
    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.weight_path or not self.weight_path.is_file():
                raise FileNotFoundError(self.unavailable_reason or "YOLO 权重未配置")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "未安装 ultralytics。YOLO 是可选依赖（AGPL-3.0），"
                    "请安装 requirements-yolo.txt 后重试。"
                ) from exc

            try:
                import torch

                self.device = "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

            model = YOLO(str(self.weight_path))
            names = getattr(model, "names", {}) or {}
            self._names = {int(k): str(v) for k, v in dict(names).items()}
            # 预热：把 CUDA context / 内核编译放到加载阶段，避免第一次对比被冷启动污染
            try:
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                model.predict(
                    dummy,
                    imgsz=self.imgsz,
                    conf=0.99,
                    device=self.device,
                    verbose=False,
                )
                self._warm = True
            except Exception:
                pass
            self._model = model
            self._ready = True
            return model

    def warmup(self) -> None:
        if not self.available:
            return
        try:
            self._load()
        except Exception:
            pass

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "available": self.available,
            "reason": self.unavailable_reason or None,
            "byom": True,
            "model": self.weight_path.name if self.weight_path else None,
            "path": str(self.weight_path) if self.weight_path else None,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "hint": "docs/yolo-byom.md",
        }

    def _predict(self, img: np.ndarray) -> list[_Obj]:
        model = self._load()
        with self._lock:
            result = model.predict(
                img,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )[0]
        out: list[_Obj] = []
        if result.boxes is None:
            return out
        xyxy = result.boxes.xyxy.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        h, w = img.shape[:2]
        for (x0, y0, x1, y1), cid, score in zip(xyxy, clss, confs):
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w - 1, int(x1)), min(h - 1, int(y1))
            if x1 - x0 < 5 or y1 - y0 < 5:
                continue
            out.append(
                _Obj(
                    int(cid),
                    self._names.get(int(cid), str(int(cid))),
                    float(score),
                    x0,
                    y0,
                    x1,
                    y1,
                )
            )
        return out

    # ---------------------------------------------------------------- 比对
    def _crop_diff(
        self, before: np.ndarray, after: np.ndarray, box: _Obj
    ) -> tuple[float, int, int]:
        pad = 2
        h, w = before.shape[:2]
        x0, y0 = max(0, box.x0 - pad), max(0, box.y0 - pad)
        x1, y1 = min(w, box.x1 + pad), min(h, box.y1 + pad)
        cb = cv2.cvtColor(before[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        ca = cv2.cvtColor(after[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        if cb.size == 0 or ca.size == 0:
            return 0.0, 0, 0
        diff = cv2.absdiff(cb, ca)
        changed = int(np.count_nonzero(diff > 18))
        total = int(diff.size)
        return (changed / total if total else 0.0), changed, total

    def detect(self, before: np.ndarray, after: np.ndarray) -> MethodResult:
        if not self.available:
            reason = self.unavailable_reason or "YOLO 权重未配置"
            return MethodResult(
                key=self.key,
                name=self.name,
                ok=False,
                error="YOLO 权重未配置（BYOM）",
                summary=reason,
                details={
                    "unavailable": True,
                    "reason": reason,
                    "docs": "docs/yolo-byom.md",
                },
            )

        before_objs = self._predict(before)
        after_objs = self._predict(after)

        # 同类目标按 IoU 贪心配对
        candidates: list[tuple[float, int, int]] = []
        for i, b in enumerate(before_objs):
            for j, a in enumerate(after_objs):
                if b.cls_id != a.cls_id:
                    continue
                iou = _iou(b, a)
                if iou >= 0.3:
                    candidates.append((iou, i, j))
        candidates.sort(reverse=True)
        used_b: set[int] = set()
        used_a: set[int] = set()
        pairs: list[tuple[int, int, float]] = []
        for iou, i, j in candidates:
            if i in used_b or j in used_a:
                continue
            used_b.add(i)
            used_a.add(j)
            pairs.append((i, j, iou))

        changed: list[dict[str, Any]] = []
        for i, j, iou in pairs:
            b, a = before_objs[i], after_objs[j]
            ratio, changed_px, total = self._crop_diff(before, after, b)
            if (ratio > 0.002 and changed_px >= 6) or (changed_px >= 50 and ratio > 0.0008):
                changed.append(
                    {
                        "cls": a.cls_name,
                        "score": round(a.score, 4),
                        "iou": round(iou, 3),
                        "box": [a.x0, a.y0, a.x1 - a.x0, a.y1 - a.y0],
                        "changedPixels": changed_px,
                        "cropPixels": total,
                        "diffRatio": round(ratio, 6),
                    }
                )

        # 只在一边出现的框：也做同坐标裁剪差，避免 NMS 抖动被当成“新增目标”
        for idx in [i for i in range(len(before_objs)) if i not in used_b]:
            box = before_objs[idx]
            ratio, changed_px, total = self._crop_diff(before, after, box)
            if ratio > 0.02 and changed_px >= 12:
                changed.append(
                    {
                        "cls": box.cls_name,
                        "score": round(box.score, 4),
                        "iou": 0.0,
                        "box": [box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0],
                        "changedPixels": changed_px,
                        "cropPixels": total,
                        "diffRatio": round(ratio, 6),
                        "oneSided": "before",
                    }
                )
        for idx in [j for j in range(len(after_objs)) if j not in used_a]:
            box = after_objs[idx]
            ratio, changed_px, total = self._crop_diff(before, after, box)
            if ratio > 0.02 and changed_px >= 12:
                changed.append(
                    {
                        "cls": box.cls_name,
                        "score": round(box.score, 4),
                        "iou": 0.0,
                        "box": [box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0],
                        "changedPixels": changed_px,
                        "cropPixels": total,
                        "diffRatio": round(ratio, 6),
                        "oneSided": "after",
                    }
                )

        has_change = bool(changed)
        if has_change:
            max_ratio = max(c["diffRatio"] for c in changed)
            max_score = max(c["score"] for c in changed)
            confidence = clamp(0.5 + max_score * 0.3 + min(0.2, max_ratio * 20))
            summary = (
                f"在 {len(changed)} 个 UI 元素内部检测到像素变化"
                f"（最大变化占比 {max_ratio * 100:.2f}%）。"
                "YOLO 负责把范围缩到具体元素，变化是新增还是消失仍需颜色/人工判断。"
            )
        else:
            confidence = 0.55 if (before_objs or after_objs) else 0.3
            if before_objs or after_objs:
                summary = (
                    f"共检测到 {len(before_objs)} / {len(after_objs)} 个 UI 目标，"
                    "所有已框住的元素内部都没有超过阈值的变化。"
                )
            else:
                summary = "没有检测到可用的 UI 目标，YOLO 无法给出有效判断（可能是模型不适配这张图）。"

        regions: list[Region] = []
        for c in changed[:14]:
            x, y, w, h = c["box"]
            regions.append(
                Region(
                    x,
                    y,
                    w,
                    h,
                    f"{c['cls']} · 变 {c['diffRatio'] * 100:.1f}%",
                    "changed",
                    score=c["diffRatio"],
                )
            )
        # 未变化目标只在“确实有元素内变化”时画进叠加层作参考，
        # 不计入“证据区域”；无变化时保持原图干净。
        context: list[Region] = list(regions)
        if has_change:
            changed_boxes = {(r.x, r.y, r.w, r.h) for r in regions}
            for o in after_objs[:40]:
                if (o.x0, o.y0, o.x1 - o.x0, o.y1 - o.y0) not in changed_boxes:
                    context.append(Region(o.x0, o.y0, o.x1 - o.x0, o.y1 - o.y0, "", "object"))

        avg_iou = sum(p[2] for p in pairs) / len(pairs) if pairs else 0.0
        metrics = [
            Metric("beforeObjects", "前图目标", len(before_objs), "个"),
            Metric("afterObjects", "后图目标", len(after_objs), "个"),
            Metric("matched", "匹配对", len(pairs), "对"),
            Metric("changedObjects", "变化目标", len(changed), "个"),
            Metric("avgIou", "平均 IoU", round(avg_iou, 3), ""),
        ]
        details = {
            "model": self.weight_path.name if self.weight_path else None,
            "modelPath": str(self.weight_path) if self.weight_path else None,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "beforeObjects": [o.to_dict() for o in before_objs[:80]],
            "afterObjects": [o.to_dict() for o in after_objs[:80]],
            "matchedPairs": len(pairs),
            "changed": changed,
            "note": "只比较模型框内的区域；红点若落在检测框外，会漏检。",
        }

        overlay = render_overlay(
            after,
            context,
            notes=[
                f"YOLO · 目标 {len(before_objs)} → {len(after_objs)} · 匹配 {len(pairs)} 对",
                f"元素内变化 {len(changed)} 个 · {self.device} · {self.weight_path.name if self.weight_path else '-'}",
            ],
        )

        return MethodResult(
            key=self.key,
            name=self.name,
            has_change=has_change,
            change_type="content" if has_change else "none",
            confidence=confidence,
            summary=summary,
            regions=regions,
            metrics=metrics,
            details=details,
            overlay=overlay,
        )
