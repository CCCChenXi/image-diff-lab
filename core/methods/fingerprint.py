"""方法 6：灰度指纹差分（dHash / aHash / pHash + 分块灰度差）。

回答：**两张图还是不是“同一张画面”？**
做法：把灰度图缩到极小尺寸，算感知哈希（指纹），比较汉明距离；
同时用 16×16 分块灰度差给出一个很粗的“哪里变了”的热力图。

它和 SSIM 的区别：
  - 指纹是“内容是否同一画面”的快速判据，对缩放、JPEG 压缩更稳；
  - SSIM 是逐窗口结构相似度，更细，但仍是全局指标。
两者都抓不住几十像素的红点 —— 这是全局方法的共同盲区。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging import render_overlay
from ..models import MethodResult, Metric, Region
from .base import ChangeMethod, clamp


def _bits_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.astype(np.uint8) != b.astype(np.uint8)))


class GrayFingerprintMethod(ChangeMethod):
    key = "gray_fingerprint"
    name = "灰度指纹差分"
    description = "dHash / aHash / pHash 感知指纹 + 16×16 分块灰度差。"
    color = "#b7791f"
    cost = "毫秒级"
    strength = "极快判断是否同一画面，对缩放/压缩稳健"
    blind_spot = "全局指纹，红点这类局部小改动会被缩图抹平"
    order = 6

    hash_bits = 64
    hash_threshold = 6          # 汉明距离超过 6/64 位视为整体变化
    block_grid = 16             # 16×16 分块
    block_threshold = 32        # 最大块差超过 32/255 视为整体变化

    def _fingerprints(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # dHash：9×8 的横向梯度，64 bit，对亮度渐变不敏感
        d = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        d_bits = (d[:, 1:] > d[:, :-1]).reshape(-1)

        # aHash：8×8 与均值比较，64 bit，最便宜
        a = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        a_bits = (a > float(a.mean())).reshape(-1)

        # pHash：32×32 DCT 左上 8×8 低频，64 bit，对压缩更稳
        p = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(p)[:8, :8]
        med = float(np.median(dct[1:]))
        p_bits = (dct > med).reshape(-1)

        return d_bits, a_bits, p_bits

    def _block_diff(self, gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        grid = self.block_grid
        b1 = cv2.resize(gray1, (grid, grid), interpolation=cv2.INTER_AREA).astype(np.float32)
        b2 = cv2.resize(gray2, (grid, grid), interpolation=cv2.INTER_AREA).astype(np.float32)
        return np.abs(b1 - b2)

    def detect(self, before: np.ndarray, after: np.ndarray) -> MethodResult:
        g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        d1, a1, p1 = self._fingerprints(g1)
        d2, a2, p2 = self._fingerprints(g2)
        d_dist = _hamming(d1, d2)
        a_dist = _hamming(a1, a2)
        p_dist = _hamming(p1, p2)
        # dHash 与 pHash 取较大值：任一指纹认为“不是同一画面”就保留证据
        main_dist = max(d_dist, p_dist)

        block = self._block_diff(g1, g2)
        block_max = float(block.max()) if block.size else 0.0
        has_change = main_dist > self.hash_threshold or block_max > self.block_threshold

        h, w = g1.shape[:2]
        regions: list[Region] = []
        if has_change:
            cell_w = w / self.block_grid
            cell_h = h / self.block_grid
            cut = max(24.0, block_max * 0.5)
            cells = [
                (float(block[y, x]), x, y)
                for y in range(self.block_grid)
                for x in range(self.block_grid)
                if block[y, x] >= cut
            ]
            cells.sort(reverse=True)
            for value, x, y in cells[:8]:
                regions.append(
                    Region(
                        int(x * cell_w),
                        int(y * cell_h),
                        max(1, int(cell_w)),
                        max(1, int(cell_h)),
                        f"块差 {value:.0f}",
                        "changed",
                        score=value / 255.0,
                    )
                )

        if has_change:
            confidence = clamp(
                0.55
                + min(0.3, main_dist / max(1, self.hash_bits) * 3.0)
                + min(0.12, block_max / 255.0 * 0.3)
            )
            summary = (
                f"dHash 距离 {d_dist}/64、pHash 距离 {p_dist}/64，"
                f"最大 16×16 块差 {block_max:.0f}/255，判定整体画面发生变化。"
                "指纹法适合快速判断“是否同一画面”，定位只是粗粒度分块。"
            )
        else:
            confidence = clamp(0.94 - min(0.35, main_dist / 20.0))
            summary = (
                f"dHash 距离 {d_dist}/64、pHash 距离 {p_dist}/64，"
                f"最大块差 {block_max:.0f}/255，指纹一致。"
                "该方法对红点这类局部小改动不敏感，适合排除“整页换图/滚动”。"
            )

        similarity = 1.0 - main_dist / self.hash_bits
        metrics = [
            Metric("dHash", "dHash 距离", d_dist, "位"),
            Metric("pHash", "pHash 距离", p_dist, "位"),
            Metric("similarity", "指纹相似度", round(similarity * 100, 2), "%"),
            Metric("blockMax", "最大块差", round(block_max, 1), "/255"),
        ]
        details = {
            "hashBits": self.hash_bits,
            "hashThreshold": self.hash_threshold,
            "blockGrid": self.block_grid,
            "blockThreshold": self.block_threshold,
            "dHash": {"before": _bits_hex(d1), "after": _bits_hex(d2), "distance": d_dist},
            "aHash": {"before": _bits_hex(a1), "after": _bits_hex(a2), "distance": a_dist},
            "pHash": {"before": _bits_hex(p1), "after": _bits_hex(p2), "distance": p_dist},
            "blockMax": round(block_max, 3),
            "regions": len(regions),
            "note": "aHash 仅作参考，不参与判定；dHash/pHash 取较大距离。",
        }

        heat = cv2.resize(block, (w, h), interpolation=cv2.INTER_NEAREST)
        heat = np.clip(heat, 0, 255).astype(np.uint8)
        overlay = render_overlay(
            after,
            regions if has_change else (),
            heat=heat if has_change else None,
            notes=[
                f"灰度指纹 · dHash {d_dist}/64 · pHash {p_dist}/64",
                f"最大块差 {block_max:.0f}/255 · {'判定变化' if has_change else '判定未变化'}",
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
