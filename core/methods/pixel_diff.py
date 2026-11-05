"""方法 1：OpenCV 像素差分。

回答：**哪些区域变了？**
做法：灰度绝对差 → 阈值 → 形态学去噪 → 连通域。它不知道“为什么变”，
所以类型只能给 `content`（内容变化）；新增/消失要靠 HSV 或人工判断。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging import render_overlay
from ..models import MethodResult, Metric, Region
from .base import ChangeMethod, clamp, ratio_text


class PixelDiffMethod(ChangeMethod):
    key = "pixel_diff"
    name = "OpenCV 图像差异"
    description = "灰度绝对差 + 阈值分割 + 连通域，定位变化区域。"
    color = "#e76f51"
    cost = "毫秒级"
    strength = "定位准确、速度最快"
    blind_spot = "只回答“哪里变了”，不知道是新增还是消失"
    order = 1

    threshold = 22
    min_area = 8

    def detect(self, before: np.ndarray, after: np.ndarray) -> MethodResult:
        gb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        ga = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        gb = cv2.GaussianBlur(gb, (3, 3), 0)
        ga = cv2.GaussianBlur(ga, (3, 3), 0)
        diff = cv2.absdiff(gb, ga)

        raw_mask = (diff > self.threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = raw_mask.shape[:2]
        regions: list[Region] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, rw, rh = cv2.boundingRect(contour)
            regions.append(
                Region(
                    x=x,
                    y=y,
                    w=rw,
                    h=rh,
                    label=f"差异 {len(regions) + 1} · {int(area)}px",
                    kind="changed",
                    score=round(min(1.0, area / max(1.0, h * w * 0.02)), 4),
                )
            )
            if len(regions) >= 12:
                break

        changed_pixels = int(cv2.countNonZero(raw_mask))
        total = int(h * w)
        changed_ratio = changed_pixels / total if total else 0.0
        has_change = bool(regions)
        peak = int(np.percentile(diff[raw_mask > 0], 95)) if changed_pixels else 0

        if has_change:
            confidence = clamp(0.52 + (peak / 255.0) * 0.32 + min(0.12, changed_ratio * 400))
            summary = (
                f"检测到 {len(regions)} 处差异，共 {changed_pixels} 像素"
                f"（{ratio_text(changed_pixels, total)}）。"
                "像素差分能定位变化，但无法判断红点是新增还是消失。"
            )
        else:
            confidence = clamp(0.97 - min(0.35, changed_ratio * 3000))
            summary = f"未检测到超过阈值（{self.threshold}）的像素差异。"

        metrics = [
            Metric("changedPixels", "变化像素", changed_pixels, "px"),
            Metric("changedRatio", "变化占比", round(changed_ratio * 100, 4), "%"),
            Metric("regions", "连通区域", len(regions), "处"),
            Metric("peakDiff", "最大差异", peak, "/255"),
        ]
        details = {
            "threshold": self.threshold,
            "minArea": self.min_area,
            "changedPixels": changed_pixels,
            "changedRatio": round(changed_ratio, 8),
            "regionCount": len(regions),
            "maxDiff": int(diff.max()) if diff.size else 0,
            "p95Diff": peak,
            "note": "类型固定为 content：灰度差分没有颜色语义。",
        }

        overlay = render_overlay(
            after,
            regions,
            heat=np.where(raw_mask > 0, diff, 0).astype(np.uint8),
            notes=[
                f"像素差分 · 阈值 {self.threshold} · 变化 {ratio_text(changed_pixels, total)}",
                f"{len(regions)} 处区域 · 峰值差 {peak}/255",
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
