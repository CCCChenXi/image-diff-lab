"""方法 3：SSIM 结构相似度。

回答：**两张图的整体结构变了多少？**
SSIM 是全局指标：局部 100 像素的红点会被整张图的均值稀释，所以它通常
判不出“小红点消失”。这也正是它在实验里的角色 —— 辅助，而不是主判据。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging import render_overlay
from ..models import MethodResult, Metric, Region
from .base import ChangeMethod, clamp


def _ssim(gray1: np.ndarray, gray2: np.ndarray) -> tuple[float, np.ndarray]:
    x = gray1.astype(np.float64)
    y = gray2.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(x, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(y, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = cv2.filter2D(x * x, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(y * y, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu1_mu2

    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean()), ssim_map


class SsimMethod(ChangeMethod):
    key = "ssim"
    name = "SSIM 结构相似度"
    description = "滑动窗口计算亮度 / 对比度 / 结构相似度，衡量整体变化。"
    color = "#457b9d"
    cost = "几十毫秒"
    strength = "对整体布局、滚动、主题切换敏感"
    blind_spot = "局部小变化会被全图平均掉，基本抓不到单个红点"
    order = 3

    threshold = 0.995

    def detect(self, before: np.ndarray, after: np.ndarray) -> MethodResult:
        gb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        ga = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        score, ssim_map = _ssim(gb, ga)
        has_change = score < self.threshold

        regions: list[Region] = []
        if has_change:
            # 只在整体已判定变化时再找“最不相似”的局部，避免给未变化图制造假框
            bad = (ssim_map < 0.55).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bad = cv2.morphologyEx(bad, cv2.MORPH_OPEN, kernel)
            bad = cv2.dilate(bad, kernel, iterations=2)
            contours, _ = cv2.findContours(bad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = max(120, int(ssim_map.size * 0.0004))
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                if cv2.contourArea(contour) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                # ssim_map 因 11x11 窗口裁边，坐标整体偏移 5px
                regions.append(Region(x + 5, y + 5, w, h, "低相似区域", "changed"))
                if len(regions) >= 6:
                    break

        if has_change:
            confidence = clamp(0.55 + (self.threshold - score) / 0.18 * 0.4)
            summary = (
                f"整体 SSIM = {score:.5f}，低于阈值 {self.threshold}，"
                "判定整体结构发生变化。"
            )
        else:
            confidence = clamp(0.6 + (score - self.threshold) * 220, 0.6, 0.95)
            summary = (
                f"整体 SSIM = {score:.5f}，高于阈值 {self.threshold}，判定未变化。"
                "局部小红点被全局平均稀释，是 SSIM 的典型盲区。"
            )

        diff_score = max(0.0, 1.0 - score)
        local_min = float(ssim_map.min()) if ssim_map.size else 1.0
        metrics = [
            Metric("ssim", "SSIM 分数", round(score, 6), ""),
            Metric("dssim", "差异度 1-SSIM", round(diff_score, 6), ""),
            Metric("threshold", "判定阈值", self.threshold, ""),
            Metric("localMin", "最低局部 SSIM", round(local_min, 4), ""),
        ]
        details = {
            "ssim": round(score, 8),
            "threshold": self.threshold,
            "regionCount": len(regions),
            "localMin": round(local_min, 6),
            "note": "SSIM 阈值越低越敏感；0.995 适合截图整体对比，不适合检测几十像素的红点。",
        }

        heat = None
        if has_change:
            heat = np.clip((1.0 - ssim_map) * 255.0 * 6.0, 0, 255).astype(np.uint8)
        overlay = render_overlay(
            after,
            regions,
            heat=heat,
            notes=[
                f"SSIM {score:.5f} · 阈值 {self.threshold}",
                f"最低局部 {local_min:.3f} · {'判定变化' if has_change else '判定未变化'}",
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
