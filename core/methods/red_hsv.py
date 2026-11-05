"""方法 2：HSV 红色检测 —— 最贴合“红点”场景的方法。

回答：**红点存在吗？有几个？在哪？是新增、消失还是移位？**
做法：HSV 双区间红色掩码 → 过滤圆形小色块 → 两图色块配对。
它不关心红点以外的任何变化，所以对“联系人旁边的红点”最直接。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..imaging import render_overlay
from ..models import MethodResult, Metric, Region
from .base import ChangeMethod, clamp


@dataclass
class _Blob:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float

    @property
    def diag(self) -> float:
        return max(6.0, math.hypot(self.w, self.h))

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "area": self.area,
            "center": [round(self.cx, 1), round(self.cy, 1)],
        }


class HsvRedMethod(ChangeMethod):
    key = "red_hsv"
    name = "HSV 红色检测"
    description = "HSV 双区间红色掩码 + 圆形小块过滤，比较红点数量与位置。"
    color = "#e63946"
    cost = "毫秒级"
    strength = "直接给出新增 / 消失 / 移位"
    blind_spot = "只认红色；红色头像、红包等红色元素会进入候选"
    order = 2

    s_min = 85
    v_min = 60
    min_area = 6
    max_area = 2400
    min_extent = 0.30
    match_distance = 1.0     # 归一化中心距阈值（除以色块对角线）

    def _red_mask(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        low = cv2.inRange(hsv, (0, self.s_min, self.v_min), (10, 255, 255))
        high = cv2.inRange(hsv, (168, self.s_min, self.v_min), (180, 255, 255))
        mask = cv2.bitwise_or(low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _blobs(self, mask: np.ndarray) -> list[_Blob]:
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out: list[_Blob] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.min_area or area > self.max_area:
                continue
            if w < 3 or h < 3:
                continue
            aspect = w / float(h)
            extent = area / float(w * h)
            if not (0.42 <= aspect <= 2.4):
                continue
            if extent < self.min_extent:
                continue
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            out.append(_Blob(x, y, w, h, area, cx, cy))
        out.sort(key=lambda b: (-b.area, b.cy, b.cx))
        return out

    def _match(
        self, before: list[_Blob], after: list[_Blob]
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        pairs: list[tuple[int, int, float]] = []
        used_after: set[int] = set()
        for i, b in enumerate(before):
            best_j = -1
            best_cost = 1e9
            for j, a in enumerate(after):
                if j in used_after:
                    continue
                dist = math.hypot(b.cx - a.cx, b.cy - a.cy)
                cost = dist / max(b.diag, a.diag)
                if cost <= self.match_distance and cost < best_cost:
                    best_cost = cost
                    best_j = j
            if best_j >= 0:
                used_after.add(best_j)
                pairs.append((i, best_j, best_cost))
        used_before = {i for i, _, _ in pairs}
        return (
            pairs,
            [i for i in range(len(before)) if i not in used_before],
            [j for j in range(len(after)) if j not in used_after],
        )

    def detect(self, before: np.ndarray, after: np.ndarray) -> MethodResult:
        mb = self._red_mask(before)
        ma = self._red_mask(after)
        bb = self._blobs(mb)
        aa = self._blobs(ma)

        pairs, unmatched_b, unmatched_a = self._match(bb, aa)
        moved: list[tuple[_Blob, _Blob, float]] = []
        added: list[_Blob] = []
        removed: list[_Blob] = []

        # 同尺寸色块但中心明显偏移 → 同一个小红点移位
        for i, j, _cost in pairs:
            b, a = bb[i], aa[j]
            dist = math.hypot(b.cx - a.cx, b.cy - a.cy)
            if dist > 8:
                moved.append((b, a, dist))

        # 精确匹配不上的：若前后各剩一个且面积接近，按“移位”处理（跨行的红点）
        rest_b = list(unmatched_b)
        rest_a = list(unmatched_a)
        if len(rest_b) == len(rest_a) and rest_b:
            for i in list(rest_b):
                b = bb[i]
                best_j = None
                best_ratio = 1e9
                for j in rest_a:
                    a = aa[j]
                    area_ratio = max(a.area / b.area, b.area / a.area)
                    if area_ratio < 2.6 and area_ratio < best_ratio:
                        best_ratio = area_ratio
                        best_j = j
                if best_j is not None:
                    a = aa[best_j]
                    moved.append((b, a, math.hypot(b.cx - a.cx, b.cy - a.cy)))
                    rest_b.remove(i)
                    rest_a.remove(best_j)
        added = [aa[j] for j in rest_a]
        removed = [bb[i] for i in rest_b]

        has_change = bool(added or removed or moved)
        if not has_change:
            if bb or aa:
                change_type = "none"
                confidence = 0.92 if len(bb) == len(aa) == len(pairs) else 0.78
                summary = (
                    f"两张图都检测到 {len(aa)} 个红色小块，数量与位置一致，"
                    "未发现红点新增、消失或移位。"
                )
            else:
                change_type = "none"
                confidence = 0.78
                summary = "两张图都没有检测到红色小色块，判定红点未出现。"
        elif added and not removed and not moved:
            change_type = "added"
            confidence = clamp(0.9 + min(0.07, len(added) * 0.02))
            summary = f"后图新增了 {len(added)} 个红点，前图对应位置没有红色目标。"
        elif removed and not added and not moved:
            change_type = "removed"
            confidence = clamp(0.9 + min(0.07, len(removed) * 0.02))
            summary = f"前图的 {len(removed)} 个红点在后图消失。"
        elif moved and not added and not removed:
            change_type = "moved"
            confidence = 0.88
            d = moved[0][2]
            summary = f"红点数量未变，但位置移动了约 {d:.0f}px，判定为位置变化。"
        else:
            change_type = "content"
            confidence = 0.66
            summary = (
                f"红色目标同时存在新增 {len(added)}、消失 {len(removed)}、"
                f"移位 {len(moved)}，属于混合变化。"
            )

        regions: list[Region] = []
        arrows: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
        for a in added:
            regions.append(Region(a.x, a.y, a.w, a.h, "新增红点", "added"))
        for b in removed:
            regions.append(Region(b.x, b.y, b.w, b.h, "红点消失", "removed"))
        for b, a, dist in moved:
            regions.append(
                Region(
                    a.x,
                    a.y,
                    a.w,
                    a.h,
                    f"移位 {a.cx - b.cx:+.0f},{a.cy - b.cy:+.0f}px",
                    "moved",
                )
            )
            arrows.append(
                (int(b.cx), int(b.cy), int(a.cx), int(a.cy), (244, 162, 97))
            )

        metrics = [
            Metric("beforeCount", "前图红点", len(bb), "个"),
            Metric("afterCount", "后图红点", len(aa), "个"),
            Metric("added", "新增", len(added), "个"),
            Metric("removed", "消失", len(removed), "个"),
            Metric("moved", "移位", len(moved), "个"),
        ]
        details = {
            "hsvRange": [[0, self.s_min, self.v_min], [10, 255, 255], [168, self.s_min, self.v_min], [180, 255, 255]],
            "blobFilter": {
                "area": [self.min_area, self.max_area],
                "aspect": [0.42, 2.4],
                "extent": self.min_extent,
            },
            "before": [b.to_dict() for b in bb],
            "after": [a.to_dict() for a in aa],
            "matchedPairs": len(pairs),
            "added": [b.to_dict() for b in added],
            "removed": [b.to_dict() for b in removed],
            "moved": [
                {"from": b.to_dict(), "to": a.to_dict(), "distance": round(d, 1)}
                for b, a, d in moved
            ],
            "note": "红点判定基于颜色与形状；红色头像/红包等可能进入候选，但两张图中不变的红点会配对忽略。",
        }

        overlay = render_overlay(
            after,
            regions,
            mask=ma,
            mask_color=(0, 220, 255),
            arrows=arrows,
            notes=[
                f"HSV 红色检测 · 前 {len(bb)} 个 / 后 {len(aa)} 个",
                f"新增 {len(added)} · 消失 {len(removed)} · 移位 {len(moved)}",
            ],
        )

        return MethodResult(
            key=self.key,
            name=self.name,
            has_change=has_change,
            change_type=change_type,
            confidence=confidence,
            summary=summary,
            regions=regions,
            metrics=metrics,
            details=details,
            overlay=overlay,
        )
