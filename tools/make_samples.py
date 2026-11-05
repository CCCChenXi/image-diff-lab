"""生成内置示例图对：从真实微信截图里“画/擦”一个红点。

只用于页面上的「载入示例」，不是评测集 —— 项目按需求只做单次对比，
不计算数据集准确率。示例对的红点位置是已知的，方便肉眼核对方法输出。

    ..\\wechat-vision\\.venv\\Scripts\\python.exe tools\\make_samples.py --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
WECHAT = ROOT.parent / "wechat-vision"
IMAGES = WECHAT / "data" / "dataset" / "images" / "train"
OUT = ROOT / "web" / "samples"

BASE_CANDIDATES = [
    "light_chat_r2_full_session01_014_chat_1280x860.png",
    "light_chat_r1_full_session06_019_chat_800x900.png",
    "acc4_chat_full_base_001_chat_661x933.png",
]

# 检测不到 session_item 时的兜底几何（根据本机截图实测）
FALLBACK_ROWS = [(60, 80, 296, 145), (59, 146, 297, 210), (59, 211, 296, 275)]


def read_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不了图片：{path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(f"编码失败：{path}")
    buf.tofile(str(path))


def detect_rows(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        from ultralytics import YOLO

        weights = WECHAT / "training" / "runs" / "yolo26n_960_v3" / "weights" / "best.pt"
        if not weights.is_file():
            return FALLBACK_ROWS
        model = YOLO(str(weights))
        result = model.predict(img, imgsz=960, conf=0.25, device="cpu", verbose=False)[0]
        rows = []
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
            if result.names[int(cls)] == "session_item":
                x0, y0, x1, y1 = (int(v) for v in box)
                rows.append((x0, y0, x1, y1))
        rows.sort(key=lambda r: r[1])
        return rows or FALLBACK_ROWS
    except Exception as exc:
        print(f"⚠️ YOLO 检测失败（{exc}），用兜底坐标")
        return FALLBACK_ROWS


def draw_dot(img: np.ndarray, row: tuple[int, int, int, int]) -> np.ndarray:
    """在会话行头像右上角画一个无数字的微信风格红点。"""
    x0, y0, _x1, _y1 = row
    h = row[3] - row[1]
    r = max(5, int(round(h * 0.11)))
    cx = x0 + int(round(h * 0.74))
    cy = y0 + int(round(h * 0.22))
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=(250, 81, 81),           # 微信红 #FA5151
        outline=(255, 255, 255),
        width=2,
    )
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--base", default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    base_path = None
    if args.base:
        p = Path(args.base)
        if p.is_file():
            base_path = p
    if base_path is None:
        for name in BASE_CANDIDATES:
            p = IMAGES / name
            if p.is_file():
                base_path = p
                break
    if base_path is None:
        candidates = sorted(IMAGES.glob("*.png"))[:1]
        if not candidates:
            raise SystemExit(f"没找到基础截图：{IMAGES}")
        base_path = candidates[0]

    base = read_image(base_path)
    rows = detect_rows(base)
    print(f"基础图：{base_path.name}  {base.shape[1]}x{base.shape[0]}  会话行 {len(rows)}")

    r1 = rows[0]
    r2 = rows[min(2, len(rows) - 1)]
    before_added = base
    after_added = draw_dot(base, r1)
    before_removed = draw_dot(base, r1)
    after_moved = draw_dot(base, r2)

    pairs = [
        ("added", "红点新增", "第二张图在第一个联系人旁多了一个红点。", "added", base, after_added),
        ("removed", "红点消失", "第一张图有红点，第二张图红点消失。", "removed", before_removed, base),
        ("moved", "红点移位", "红点从第一个联系人的头像移到第三个联系人。", "moved", before_removed, after_moved),
        ("none", "无变化", "两张图完全相同，用来观察方法会不会误报。", "none", base, base),
    ]

    meta = []
    for pid, label, note, expect, before, after in pairs:
        before_name = f"{pid}-before.png"
        after_name = f"{pid}-after.png"
        before_path, after_path = OUT / before_name, OUT / after_name
        if args.force or not before_path.is_file():
            write_image(before_path, before)
        if args.force or not after_path.is_file():
            write_image(after_path, after)
        meta.append(
            {
                "id": pid,
                "label": label,
                "note": note,
                "expect": expect,
                "before": before_name,
                "after": after_name,
                "base": base_path.name,
                "row": list(r1 if pid != "moved" else r2),
            }
        )
        print(f"  ✓ {label}: {before_name} → {after_name}")

    (OUT / "samples.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"示例已写入 {OUT}")


if __name__ == "__main__":
    main()
