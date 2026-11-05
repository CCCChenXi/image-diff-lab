from .base import ChangeMethod
from .fingerprint import GrayFingerprintMethod
from .ocr import OcrMethod
from .orb_align import OrbAlignMethod
from .pixel_diff import PixelDiffMethod
from .red_hsv import HsvRedMethod
from .ssim import SsimMethod
from .yolo import YoloMethod

__all__ = [
    "ChangeMethod",
    "PixelDiffMethod",
    "HsvRedMethod",
    "SsimMethod",
    "OcrMethod",
    "YoloMethod",
    "GrayFingerprintMethod",
    "OrbAlignMethod",
]
