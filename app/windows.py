"""窗函数生成（核心逻辑，手写，不调用 scipy/numpy 的窗函数）。

三种对称窗：
- 矩形窗（rectangular / rect / boxcar）
- 汉宁窗（hann / hanning，两种叫法视为同一种）
- 汉明窗（hamming）

长度为 N 的对称窗用于 FIR 设计，下标 n = 0..N-1，关于 (N-1)/2 对称。
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .errors import FilterError

WindowFn = Callable[[int], list[float]]

# 名称 -> 规范名
_WINDOW_ALIASES: dict[str, str] = {
    "rect": "rectangular",
    "rectangular": "rectangular",
    "boxcar": "rectangular",
    "矩形窗": "rectangular",
    "hann": "hann",
    "hanning": "hann",
    "汉宁窗": "hann",
    "hamming": "hamming",
    "汉明窗": "hamming",
}

SUPPORTED_WINDOWS = ("rectangular", "hann", "hamming")


def normalize_window_name(name: str) -> str:
    """把外部传入的窗名归一化；不认识的名字直接拒绝。"""
    if not isinstance(name, str):
        raise FilterError(f"窗类型必须是字符串，收到的是 {type(name).__name__}")
    key = name.strip().lower()
    canonical = _WINDOW_ALIASES.get(key)
    if canonical is None:
        raise FilterError(
            f"不支持的窗类型 {name!r}；支持: rectangular(矩形窗)、hann/hanning(汉宁窗)、hamming(汉明窗)"
        )
    return canonical


def rectangular(length: int) -> list[float]:
    """矩形窗：w[n] = 1。"""
    if length <= 0:
        raise FilterError("窗长度必须为正整数")
    return [1.0] * length


def hann(length: int) -> list[float]:
    """汉宁窗：w[n] = 0.5 - 0.5 cos(2 pi n / (N-1))，首尾样本为 0。"""
    if length <= 0:
        raise FilterError("窗长度必须为正整数")
    if length == 1:
        return [1.0]
    return [
        0.5 - 0.5 * math.cos(2.0 * math.pi * n / (length - 1))
        for n in range(length)
    ]


def hamming(length: int) -> list[float]:
    """汉明窗：w[n] = 0.54 - 0.46 cos(2 pi n / (N-1))，首尾样本为 0.08。"""
    if length <= 0:
        raise FilterError("窗长度必须为正整数")
    if length == 1:
        return [1.0]
    return [
        0.54 - 0.46 * math.cos(2.0 * math.pi * n / (length - 1))
        for n in range(length)
    ]


_BUILDERS: dict[str, WindowFn] = {
    "rectangular": rectangular,
    "hann": hann,
    "hamming": hamming,
}


def get_window(name: str, length: int) -> list[float]:
    """按规范名取指定长度的窗；长度必须与滤波器抽头数一致。"""
    canonical = normalize_window_name(name)
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise FilterError("窗长度必须为正整数")
    return _BUILDERS[canonical](length)
