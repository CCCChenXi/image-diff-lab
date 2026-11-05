"""七种方法的共同接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import MethodResult


class ChangeMethod(ABC):
    key: str = ""
    name: str = ""
    description: str = ""
    color: str = "#888888"
    cost: str = ""
    strength: str = ""
    blind_spot: str = ""
    order: int = 99

    @abstractmethod
    def detect(self, before: np.ndarray, after: np.ndarray) -> MethodResult:
        """对已对齐的两张 BGR 图做检测。

        实现里**不要**自己计时：耗时由 AOP 切面在返回时写入 `elapsed_ms`。
        """

    def meta(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "color": self.color,
            "cost": self.cost,
            "strength": self.strength,
            "blindSpot": self.blind_spot,
            "order": self.order,
        }


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def ratio_text(num: int, den: int) -> str:
    if den <= 0:
        return "0%"
    return f"{num / den * 100:.3f}%"
