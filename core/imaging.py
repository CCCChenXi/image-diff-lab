"""图像解码、对齐、叠加层绘制。"""

from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import Region

MAX_SIDE = 1600          # 对齐后长边上限，避免 4K 截图把 CPU 推理拖死
MAX_IMAGE_BYTES = 12 * 1024 * 1024
OVERLAY_SIDE = 1280      # 返回给前端的叠加图长边上限

REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "added": (230, 57, 70),       # 红
    "removed": (58, 134, 255),    # 蓝
    "moved": (244, 162, 97),      # 橙
    "changed": (255, 107, 53),    # 橙红
    "object": (108, 117, 125),    # 灰
}

_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]
_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(10, int(size))
    if size in _font_cache:
        return _font_cache[size]
    for p in _FONT_CANDIDATES:
        if p.is_file():
            try:
                f = ImageFont.truetype(str(p), size)
                _font_cache[size] = f
                return f
            except OSError:
                continue
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f


def decode_data_url(value: str, field_name: str = "image") -> np.ndarray:
    """解析 data URL / 裸 base64，返回 BGR ndarray。

    ⚠️ 用 imdecode 而不是 imread：本项目路径在 `D:\\陈曦\\...`，
    OpenCV 的 Windows 文件读取对非 ASCII 路径会直接失败。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 为空")
    payload = value.strip()
    if payload.startswith("data:"):
        comma = payload.find(",")
        if comma < 0:
            raise ValueError(f"{field_name} 的 data URL 格式不正确")
        header, payload = payload[:comma], payload[comma + 1 :]
        if "base64" not in header.lower():
            raise ValueError(f"{field_name} 只支持 base64 data URL")
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"{field_name} base64 解码失败：{exc}") from exc
    if not raw:
        raise ValueError(f"{field_name} 解码后为空")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"{field_name} 超过 {MAX_IMAGE_BYTES // 1024 // 1024} MB，请先压缩"
        )
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{field_name} 不是可识别的图片（PNG/JPG/WebP）")
    return img


def _resize_max_side(img: np.ndarray, cap: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= cap:
        return img, 1.0
    scale = cap / float(long_side)
    out = cv2.resize(
        img,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return out, scale


def align_pair(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """把两张图放到同一坐标系。

    规则：
      1. 各自长边压到 MAX_SIDE 以内；
      2. 尺寸仍不同 → 后图缩放到前图尺寸（截图对比里前图是基准）；
      3. 返回每张图的原始尺寸与缩放比，前端提示“已对齐”。
    """
    if before is None or after is None:
        raise ValueError("图片解码失败")
    bh, bw = before.shape[:2]
    ah, aw = after.shape[:2]
    if bw < 8 or bh < 8 or aw < 8 or ah < 8:
        raise ValueError("图片尺寸过小，无法分析")

    b2, bs = _resize_max_side(before, MAX_SIDE)
    a2, as_ = _resize_max_side(after, MAX_SIDE)

    aligned = False
    if a2.shape[:2] != b2.shape[:2]:
        a2 = cv2.resize(a2, (b2.shape[1], b2.shape[0]), interpolation=cv2.INTER_AREA)
        aligned = True

    h, w = b2.shape[:2]
    ratio = (aw / ah) / (bw / bh) if bh and ah else 1.0
    warning = None
    if abs(ratio - 1.0) > 0.02:
        warning = (
            f"两张图宽高比相差 {(abs(ratio - 1) * 100):.1f}%，"
            "后图已被拉伸到前图尺寸，差异可能包含形变噪声。"
        )

    info = {
        "width": w,
        "height": h,
        "aligned": aligned or abs(bs - 1) > 1e-6 or abs(as_ - 1) > 1e-6,
        "warning": warning,
        "before": {
            "originalWidth": bw,
            "originalHeight": bh,
            "scale": round(bs, 4),
        },
        "after": {
            "originalWidth": aw,
            "originalHeight": ah,
            "scale": round(as_, 4),
            "resizedToBefore": aligned,
        },
    }
    return b2, a2, info


def encode_data_url(img: np.ndarray, quality: int = 82, max_side: int = OVERLAY_SIDE) -> str:
    out, _ = _resize_max_side(img, max_side)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("叠加图编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _fill_alpha(
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    color: tuple[int, int, int],
    alpha: int = 255,
) -> None:
    draw.rectangle(tuple(int(v) for v in box), fill=(*color, alpha))


def _chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fill: tuple[int, int, int],
    size: int,
    max_width: int,
) -> int:
    """画一个带底色的标签，返回下一行的 y。"""
    font = _font(size)
    raw = str(text)
    text = raw
    # 过长的标签截断
    while text and draw.textlength(text, font=font) > max_width - 16:
        text = text[:-1]
    if text != raw:
        text = text[:-1] + "…"
    tw = int(draw.textlength(text, font=font))
    th = size + 8
    draw.rounded_rectangle(
        (x, y, x + tw + 14, y + th),
        radius=6,
        fill=(*fill, 235),
        outline=(255, 255, 255, 180),
        width=1,
    )
    draw.text((x + 7, y + th / 2 - size * 0.62), text, font=font, fill=(255, 255, 255))
    return y + th + 5


def render_overlay(
    base: np.ndarray,
    regions: Iterable[Region] = (),
    heat: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    mask_color: tuple[int, int, int] = (0, 220, 255),
    notes: Sequence[str] = (),
    arrows: Sequence[tuple[int, int, int, int, tuple[int, int, int]]] = (),
    dim: bool = False,
) -> str:
    """把证据画到原图上，返回 JPEG data URL。

    heat  灰度热力图（0-255），半透明 JET 叠加；
    mask  二值掩码，用 mask_color 半透明着色；
    notes 左上角说明文字。
    """
    canvas = base.copy()
    if heat is not None and heat.size:
        heat = np.asarray(heat)
        if heat.ndim == 3:
            heat = cv2.cvtColor(heat.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # SSIM 的 map 因 11x11 窗口裁边会比原图小 10px；统一放大回去再叠加
        if heat.shape[:2] != canvas.shape[:2]:
            heat = cv2.resize(
                heat, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(canvas, 0.62, colored, 0.38, 0)
        # 只在“真正有差异”的像素上叠热力色，否则整张图会被 JET 的底色染蓝
        threshold = max(2, int(norm.max()) * 0.12)
        active = (norm > threshold).astype(np.uint8)
        active3 = cv2.merge([active, active, active])
        canvas = np.where(active3 > 0, blended, canvas)
    if mask is not None and mask.size:
        mask = np.asarray(mask)
        if mask.shape[:2] != canvas.shape[:2]:
            mask = cv2.resize(
                mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        tint = np.zeros_like(canvas)
        tint[:] = mask_color
        m = (mask > 0).astype(np.uint8)
        m = cv2.merge([m, m, m])
        canvas = np.where(m > 0, cv2.addWeighted(canvas, 0.35, tint, 0.65, 0), canvas)
    if dim:
        canvas = (canvas * 0.55).astype(np.uint8)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    h, w = canvas.shape[:2]
    base_size = max(11, min(18, int(w / 52)))
    max_label_w = max(80, int(w * 0.46))

    for r in regions:
        color = REGION_COLORS.get(r.kind, REGION_COLORS["changed"])
        x0, y0 = max(0, r.x), max(0, r.y)
        x1, y1 = min(w - 1, r.x + r.w), min(h - 1, r.y + r.h)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle((x0, y0, x1, y1), outline=(*color, 255), width=2)
        if r.label:
            _chip(draw, x0, max(0, y0 - base_size - 10), r.label, color, base_size, max_label_w)

    for x0, y0, x1, y1, color in arrows:
        draw.line((x0, y0, x1, y1), fill=(*color, 255), width=2)
        angle = math.atan2(y1 - y0, x1 - x0)
        for delta in (0.5, -0.5):
            ax = x1 - 9 * math.cos(angle + delta)
            ay = y1 - 9 * math.sin(angle + delta)
            draw.line((x1, y1, ax, ay), fill=(*color, 255), width=2)
        draw.ellipse((x0 - 3, y0 - 3, x0 + 3, y0 + 3), fill=(*color, 255))

    y = 8
    for note in notes:
        if note:
            y = _chip(draw, 8, y, note, (20, 20, 24), base_size, int(w * 0.92))

    out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return encode_data_url(out)


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
