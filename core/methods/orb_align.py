"""方法 7：ORB 特征对齐 + 残差差分。

回答：**两张图是不是只是整体位移/滚动？对齐之后还剩什么变化？**
做法：ORB 特征点 + BFMatcher + RANSAC 单应性，把后图变换到前图坐标系，
再对残差做阈值分割。它解决的是普通像素差分最怕的场景：
两张截图拍摄时窗口挪了一点、页面滚动了一点，像素差分会把整屏都判成变化。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..imaging import render_overlay
from ..models import MethodResult, Metric, Region
from .base import ChangeMethod, clamp


class OrbAlignMethod(ChangeMethod):
    key = "orb_align"
    name = "ORB 特征对齐差分"
    description = "ORB 特征 + RANSAC 单应性对齐后，再比较残差；抗窗口位移/滚动。"
    color = "#3a86ff"
    cost = "几十到一百多毫秒"
    strength = "能抵消整体平移/滚动，减少整屏误报"
    blind_spot = "特征点少或界面重复时对齐可能失败；局部小变化仍依赖残差阈值"
    order = 7

    max_features = 2000
    residual_threshold = 30
    min_area = 30

    def __init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=self.max_features, fastThreshold=10)

    def _align(
        self, before: np.ndarray, after: np.ndarray
    ) -> tuple[np.ndarray, bool, dict]:
        g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self._orb.detectAndCompute(g1, None)
        kp2, des2 = self._orb.detectAndCompute(g2, None)
        info = {
            "aligned": False,
            "keypointsBefore": len(kp1) if kp1 else 0,
            "keypointsAfter": len(kp2) if kp2 else 0,
            "matches": 0,
            "inliers": 0,
            "translation": [0.0, 0.0],
            "scale": 1.0,
            "matrix": None,
            "reason": "",
        }
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            info["reason"] = "特征点不足"
            return after, False, info

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
        good = [m for m in matches[:600] if m.distance <= 64]
        info["matches"] = len(good)
        if len(good) < 12:
            info["reason"] = f"有效匹配只有 {len(good)} 对"
            return after, False, info

        src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask is None:
            info["reason"] = "单应性估计失败"
            return after, False, info

        inliers = int(mask.sum())
        info["inliers"] = inliers
        if inliers < 8:
            info["reason"] = f"内点只有 {inliers} 个"
            return after, False, info

        H = H / H[2, 2] if abs(H[2, 2]) > 1e-9 else H
        det = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
        scale = math.sqrt(abs(det)) if det > 0 else 0.0
        tx, ty = float(H[0, 2]), float(H[1, 2])
        h, w = before.shape[:2]
        if scale < 0.5 or scale > 2.0:
            info["reason"] = f"估计缩放异常：{scale:.2f}"
            return after, False, info
        if math.hypot(tx, ty) > 0.6 * max(w, h):
            info["reason"] = f"估计位移异常：({tx:.0f}, {ty:.0f})"
            return after, False, info

        warped = cv2.warpPerspective(
            after,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        info.update(
            {
                "aligned": True,
                "translation": [round(tx, 2), round(ty, 2)],
                "scale": round(scale, 4),
                "matrix": [[round(float(v), 5) for v in row] for row in H],
                "reason": "",
            }
        )
        return warped, True, info

    def _valid_mask(self, shape: tuple[int, int], H: np.ndarray | None) -> np.ndarray:
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        if H is not None:
            mask = cv2.warpPerspective(
                mask,
                np.asarray(H, dtype=np.float64),
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.erode(mask, kernel, iterations=1)

    def detect(self, before: np.ndarray, after: np.ndarray) -> MethodResult:
        h, w = before.shape[:2]
        warped, aligned, align = self._align(before, after)
        g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        g1 = cv2.GaussianBlur(g1, (3, 3), 0)
        g2 = cv2.GaussianBlur(g2, (3, 3), 0)
        diff = cv2.absdiff(g1, g2)

        valid = self._valid_mask((h, w), align.get("matrix") if aligned else None)
        raw = (diff > self.residual_threshold).astype(np.uint8) * 255
        raw = cv2.bitwise_and(raw, valid)
        # 用 5x5 开运算去掉重采样产生的 1–2px 文字边缘，红点这类块状变化仍保留
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[Region] = []
        min_area = max(self.min_area, int(w * h * 0.00008))
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, rw, rh = cv2.boundingRect(contour)
            regions.append(
                Region(
                    x,
                    y,
                    rw,
                    rh,
                    f"残差 {len(regions) + 1} · {int(area)}px",
                    "changed",
                    score=round(min(1.0, area / max(1.0, w * h * 0.02)), 4),
                )
            )
            if len(regions) >= 12:
                break

        changed_pixels = int(cv2.countNonZero(raw))
        total = int(h * w)
        ratio = changed_pixels / total if total else 0.0
        has_change = bool(regions)

        tx, ty = align.get("translation", [0.0, 0.0])
        if has_change:
            confidence = clamp(
                0.55
                + min(0.2, align.get("inliers", 0) / max(1, align.get("matches", 1)) * 0.2)
                + min(0.2, ratio * 200)
            )
            if aligned:
                summary = (
                    f"ORB 用 {align['inliers']} 个内点估计出整体位移 "
                    f"({tx:+.1f}, {ty:+.1f})、缩放 {align['scale']:.3f}；"
                    f"对齐后仍有 {len(regions)} 处残差（{changed_pixels} px），"
                    "说明除了整体位移还有内容变化。"
                )
            else:
                summary = (
                    f"ORB 未能可靠对齐（{align.get('reason') or '内点不足'}），"
                    f"直接比较原图残差：{len(regions)} 处、{changed_pixels} px。"
                )
        else:
            confidence = 0.85 if aligned else 0.5
            if aligned:
                summary = (
                    f"ORB 用 {align['inliers']} 个内点估计出整体位移 "
                    f"({tx:+.1f}, {ty:+.1f})、缩放 {align['scale']:.3f}；"
                    "对齐后残差低于阈值，判定两张图只是整体位移/滚动，内容没有变化。"
                )
            else:
                summary = (
                    f"ORB 未能可靠对齐（{align.get('reason') or '特征不足'}），"
                    "且直接比较也没有发现超过阈值的残差。"
                )

        metrics = [
            Metric("keypoints", "特征点", f"{align['keypointsBefore']} / {align['keypointsAfter']}", ""),
            Metric("matches", "匹配对", align["matches"], "对"),
            Metric("inliers", "RANSAC 内点", align["inliers"], "个"),
            Metric("shift", "估计位移", f"{tx:+.1f}, {ty:+.1f}", "px"),
            Metric("changedPixels", "残差像素", changed_pixels, "px"),
            Metric("regions", "残差区域", len(regions), "处"),
        ]
        details = {
            "aligned": aligned,
            "reason": align.get("reason") or None,
            "keypointsBefore": align["keypointsBefore"],
            "keypointsAfter": align["keypointsAfter"],
            "matches": align["matches"],
            "inliers": align["inliers"],
            "translation": align.get("translation"),
            "scale": align.get("scale"),
            "homography": align.get("matrix"),
            "residualThreshold": self.residual_threshold,
            "changedPixels": changed_pixels,
            "changedRatio": round(ratio, 8),
            "regions": len(regions),
            "note": "对齐后残差小并不代表两张图逐像素相同：若只是页面滚动，本方法会判“无内容变化”。",
        }

        heat = np.where(raw > 0, diff, 0).astype(np.uint8)
        overlay = render_overlay(
            warped if aligned else after,
            regions,
            heat=heat if has_change else None,
            notes=[
                f"ORB 对齐 · {'成功' if aligned else '失败'} · 内点 {align['inliers']}",
                f"位移 ({tx:+.1f}, {ty:+.1f}) · 残差 {ratio * 100:.3f}%",
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
