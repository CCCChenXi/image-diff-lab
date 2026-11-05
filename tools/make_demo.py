#!/usr/bin/env python
"""生成**合成**的前后对比示例图（不包含任何真实截图）。

产物（默认写到 web/samples/）：

    demo-added-before.png / demo-added-after.png
    demo-removed-before.png / demo-removed-after.png
    demo-moved-before.png / demo-moved-after.png
    demo-none-before.png / demo-none-after.png
    samples.demo.json

页面会同时读取 samples.json（本地真实图）和 samples.demo.json（本文件生成），
所以合成示例可以和真实示例并存，不会互相覆盖。

用法：
    python tools/make_demo.py            # 已存在则跳过
    python tools/make_demo.py --force    # 重新生成
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "samples"

W, H = 900, 620
BG = (246, 247, 250)
PANEL = (255, 255, 255)
SIDE = (235, 238, 243)
LINE = (214, 219, 227)
INK = (58, 66, 82)
MUTED = (150, 158, 172)
ACCENT = (76, 132, 245)
GREEN = (46, 190, 130)
RED = (60, 60, 235)
WHITE = (255, 255, 255)

ROWS_Y = [124 + i * 92 for i in range(5)]
AVATAR_X = 134
AVATAR_R = 18


def rounded(img, x0, y0, x1, y1, color, radius=10, thickness=-1):
    cv2.rectangle(img, (x0 + radius, y0), (x1 - radius, y1), color, thickness)
    cv2.rectangle(img, (x0, y0 + radius), (x1, y1 - radius), color, thickness)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
        cv2.circle(img, (cx, cy), radius, color, thickness)


def text(img, s, x, y, scale=0.42, color=INK, bold=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold,
                cv2.LINE_AA)


def bar(img, x, y, w, h=8, color=LINE):
    rounded(img, x, y, x + w, y + h, color, radius=max(3, h // 2))


def base_frame() -> np.ndarray:
    img = np.full((H, W, 3), BG, np.uint8)

    # 顶部标题栏
    rounded(img, 24, 20, W - 24, 66, PANEL, 12)
    cv2.circle(img, (46, 43), 7, ACCENT, -1)
    text(img, "Demo App", 64, 49, 0.5, INK, 1)
    cv2.circle(img, (W - 46, 43), 5, MUTED, -1)

    # 左侧导航
    rounded(img, 24, 78, 84, H - 24, SIDE, 12)
    for i in range(4):
        cv2.circle(img, (54, 112 + i * 52), 12, ACCENT if i == 0 else MUTED,
                   -1 if i == 0 else 2)

    # 会话列表
    rounded(img, 96, 78, 356, H - 24, PANEL, 12)
    text(img, "Inbox", 112, 104, 0.48, INK, 1)
    for i, y in enumerate(ROWS_Y):
        if y + 76 > H - 36:
            break
        cv2.circle(img, (AVATAR_X, y + 24), AVATAR_R,
                   (206, 214, 228) if i else (198, 214, 240), -1)
        bar(img, 162, y + 7, 120 + (i % 3) * 18, 9, INK if i == 0 else MUTED)
        bar(img, 162, y + 28, 150 + (i % 2) * 20, 8)
        bar(img, 162, y + 48, 100, 7)
        if i == 0:
            rounded(img, 100, y - 8, 352, y + 66, (238, 243, 255), 10, 2)
        cv2.line(img, (100, y + 76), (352, y + 76), LINE, 1)

    # 右侧内容
    rounded(img, 368, 78, W - 24, H - 24, PANEL, 12)
    bar(img, 392, 96, 120, 10, INK)
    bar(img, 392, 116, 80, 7, MUTED)
    cv2.line(img, (368, 138), (W - 24, 138), LINE, 1)
    bubbles = [(400, 160, 250, 52, False), (470, 228, 300, 66, False),
               (540, 310, 230, 48, True), (500, 374, 280, 60, False),
               (560, 450, 200, 44, True)]
    for (x, y, w, h, mine) in bubbles:
        color = (232, 240, 255) if mine else (240, 242, 246)
        rounded(img, x, y, x + w, y + h, color, 12)
        bar(img, x + 18, y + 16, w - 60, 8, INK)
        bar(img, x + 18, y + 32, max(40, w - 110), 7, MUTED)
    rounded(img, 392, H - 96, W - 44, H - 44, (242, 244, 248), 14)
    bar(img, 412, H - 78, 260, 9, MUTED)
    cv2.circle(img, (W - 76, H - 70), 16, ACCENT, -1)

    return img


def draw_dot(img, row: int) -> None:
    """在某个会话行的头像右上角画红点（带白字 1）。"""
    y = ROWS_Y[row] + 24
    cx, cy = AVATAR_X + 14, y - 14
    cv2.circle(img, (cx, cy), 10, RED, -1)
    cv2.circle(img, (cx, cy), 10, WHITE, 1)
    cv2.putText(img, "1", (cx - 3, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                WHITE, 1, cv2.LINE_AA)


def pair(kind: str) -> tuple[np.ndarray, np.ndarray]:
    before = base_frame()
    after = base_frame()
    if kind == "added":
        draw_dot(after, 0)
    elif kind == "removed":
        draw_dot(before, 0)
    elif kind == "moved":
        draw_dot(before, 0)
        draw_dot(after, 2)
    elif kind != "none":
        raise ValueError(kind)
    return before, after


def imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    buf.tofile(str(path))


CASES = [
    ("demo-added", "合成 · 红点新增", "第二张图在第一个会话上多了一个红点。", "added"),
    ("demo-removed", "合成 · 红点消失", "第一张图有红点，第二张图红点消失。", "removed"),
    ("demo-moved", "合成 · 红点移位", "红点从第一个会话移到第三个会话。", "moved"),
    ("demo-none", "合成 · 无变化", "两张图完全相同，用来观察方法会不会误报。", "none"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成的前后对比示例图")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for sid, label, note, expect in CASES:
        before_name = f"{sid}-before.png"
        after_name = f"{sid}-after.png"
        before_path, after_path = OUT / before_name, OUT / after_name
        if before_path.exists() and after_path.exists() and not args.force:
            print(f"跳过（已存在）: {sid}")
        else:
            before, after = pair(expect)
            imwrite(before_path, before)
            imwrite(after_path, after)
            print(f"生成: {sid}")
        entries.append({
            "id": sid, "label": label, "note": note, "expect": expect,
            "before": before_name, "after": after_name,
        })

    meta = OUT / "samples.demo.json"
    meta.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"清单: {meta}（{len(entries)} 组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
