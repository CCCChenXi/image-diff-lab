"""方法 4：OCR 文字变化。

回答：**两张图的文字内容变了吗？**
红点不是文字，所以 OCR 对“红点消失”通常给“无变化” —— 这不是 bug，
而是它在这类场景里的天然盲区。它真正的价值是查联系人名、消息、时间等文本变化。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from ..imaging import render_overlay
from ..models import MethodResult, Metric, Region
from .base import ChangeMethod


@dataclass
class _Line:
    text: str
    norm: str
    score: float
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:60],
            "score": round(self.score, 4),
            "box": [self.x, self.y, self.w, self.h],
        }


class OcrMethod(ChangeMethod):
    key = "ocr"
    name = "OCR 文字识别"
    description = "PaddleOCR 识别两图文字，按文本内容与位置比较差异。"
    color = "#2a9d8f"
    cost = "秒级（首次含模型初始化）"
    strength = "能定位联系人名、消息、时间等文本变化"
    blind_spot = "只认文字；红点、图标、颜色变化完全看不到"
    order = 4

    def __init__(self) -> None:
        self._engine: Any = None
        self._lock = threading.Lock()
        self._ready = False
        self.init_seconds: float | None = None

    # ---------------------------------------------------------------- 模型
    def _get_engine(self) -> Any:
        with self._lock:
            if self._engine is None:
                import time

                t0 = time.perf_counter()
                # 为什么关 mkldnn：本机 Paddle 3.3 + oneDNN 在 PIR 执行器上会抛
                # `ConvertPirAttribute2RuntimeAttribute not support`，关掉后可稳定走 CPU。
                from paddleocr import PaddleOCR

                self._engine = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                )
                self.init_seconds = round(time.perf_counter() - t0, 2)
                self._ready = True
            return self._engine

    def warmup(self) -> None:
        """服务启动时预热，避免第一次对比把模型加载时间算进方法耗时。"""
        try:
            engine = self._get_engine()
            dummy = np.full((64, 320, 3), 255, dtype=np.uint8)
            with self._lock:
                engine.predict(dummy)
        except Exception:
            pass

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "engine": "PaddleOCR · PP-OCRv5 mobile (CPU)",
            "initSeconds": self.init_seconds,
        }

    # ---------------------------------------------------------------- 识别
    def _parse(self, result: Any) -> list[_Line]:
        if not result:
            return []
        r = result[0]
        texts = list(r.get("rec_texts") or [])
        scores = list(r.get("rec_scores") or [])
        boxes = r.get("rec_boxes")
        if boxes is None:
            boxes = []
        boxes = np.asarray(boxes)
        lines: list[_Line] = []
        for i, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            norm = "".join(text.split())
            if not norm:
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < 0.2:
                continue
            # 单字符低置信度碎片、纯标点行是 OCR 抖动的高发区，直接丢弃
            if len(norm) == 1 and score < 0.8:
                continue
            if not re.search(r"[\w\u4e00-\u9fff]", norm):
                continue
            if i < len(boxes):
                b = boxes[i]
                if len(b) >= 4:
                    x0, y0, x1, y1 = (float(v) for v in b[:4])
                    x, y = int(min(x0, x1)), int(min(y0, y1))
                    w, h = int(abs(x1 - x0)), int(abs(y1 - y0))
                else:
                    continue
            else:
                continue
            if w < 4 or h < 4:
                continue
            lines.append(_Line(text, norm, score, x, y, w, h))
        return lines

    def _run(self, img: np.ndarray) -> list[_Line]:
        engine = self._get_engine()
        with self._lock:
            result = engine.predict(img)
        return self._parse(result)

    # OCR 对同一张图也会有 1 个标点/字形的抖动：先精确文本，再同位置模糊匹配，
    # 只有“确实对不上”的行才算新增 / 消失。
    def _match_index(self, line: _Line, pool: list[_Line]) -> int | None:
        for i, other in enumerate(pool):
            if other.norm == line.norm:
                return i
        best_i: int | None = None
        best_ratio = 0.0
        for i, other in enumerate(pool):
            if min(len(line.norm), len(other.norm)) < 3:
                continue
            if abs(line.cx - other.cx) > max(30.0, line.w * 0.55):
                continue
            if abs(line.cy - other.cy) > max(14.0, line.h * 0.8):
                continue
            r = SequenceMatcher(None, line.norm, other.norm).ratio()
            if r > best_ratio:
                best_ratio, best_i = r, i
        return best_i if best_i is not None and best_ratio >= 0.85 else None

    def detect(self, before: np.ndarray, after: np.ndarray) -> MethodResult:
        before_lines = self._run(before)
        after_lines = self._run(after)

        pool = list(before_lines)
        added: list[_Line] = []
        for line in after_lines:
            idx = self._match_index(line, pool)
            if idx is None:
                added.append(line)
            else:
                pool.pop(idx)
        removed = pool

        has_change = bool(added or removed)
        regions: list[Region] = []
        for line in added:
            regions.append(
                Region(line.x, line.y, line.w, line.h, f"+ {line.text[:18]}", "added")
            )
        for line in removed:
            regions.append(
                Region(line.x, line.y, line.w, line.h, f"- {line.text[:18]}", "removed")
            )
        # 未变化的文字框只在“确实有文本变化”时画进叠加层作定位参考，
        # 不计入“证据区域”；无变化时保持原图干净。
        context: list[Region] = list(regions)
        if has_change:
            changed_keys = {(r.x, r.y, r.w, r.h) for r in regions}
            for line in after_lines[:80]:
                if (line.x, line.y, line.w, line.h) not in changed_keys:
                    context.append(Region(line.x, line.y, line.w, line.h, "", "object"))

        avg_score = (
            sum(line.score for line in after_lines) / len(after_lines)
            if after_lines
            else 0.0
        )
        if has_change:
            confidence = 0.78
            summary = (
                f"识别到文字变化：新增 {len(added)} 行、消失 {len(removed)} 行。"
                "OCR 只比较文本，无法说明红点等非文字元素的变化。"
            )
        elif before_lines or after_lines:
            confidence = 0.82
            summary = (
                f"文字内容一致（前 {len(before_lines)} 行 / 后 {len(after_lines)} 行），"
                "未发现文字层面的变化。红点不是文字，OCR 不会报出它。"
            )
        else:
            confidence = 0.5
            summary = "两张图都没有识别到文字，OCR 无法给出有效判断。"

        metrics = [
            Metric("beforeLines", "前图文本", len(before_lines), "行"),
            Metric("afterLines", "后图文本", len(after_lines), "行"),
            Metric("added", "新增文本", len(added), "行"),
            Metric("removed", "消失文本", len(removed), "行"),
            Metric("avgScore", "平均识别置信度", round(avg_score, 4), ""),
        ]
        details = {
            "engine": "PaddleOCR · PP-OCRv5 mobile · CPU",
            "initSeconds": self.init_seconds,
            "avgScore": round(avg_score, 4),
            "added": [line.to_dict() for line in added],
            "removed": [line.to_dict() for line in removed],
            "beforeTexts": [line.text[:60] for line in before_lines],
            "afterTexts": [line.text[:60] for line in after_lines],
            "note": "先精确文本匹配，再做同位置模糊匹配（相似度≥0.85）过滤 OCR 抖动；红点不产生文本。",
        }

        overlay = render_overlay(
            after,
            context,
            notes=[
                f"OCR · 前 {len(before_lines)} 行 / 后 {len(after_lines)} 行",
                f"文本新增 {len(added)} · 消失 {len(removed)}",
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
