"""相位展开（核心逻辑手写）。

主值相位 atan2 落在 (-pi, pi]，相邻频点跨越支割时会出现 +/- 2pi 的跳变。
展开规则：若第 i 点相对前一点的跳变大于 +pi，则后续整体减 2pi；
小于 -pi 则整体加 2pi；跳变恰为 pi 时不处理。
"""

from __future__ import annotations

import math


def unwrap_phase(phases: list[float], period: float = 2.0 * math.pi) -> list[float]:
    """对按频率升序排列的主值相位做一维相位展开。"""
    if not phases:
        return []
    half = period / 2.0
    unwrapped = [phases[0]]
    offset = 0.0
    for prev, current in zip(phases, phases[1:]):
        delta = current - prev
        if delta > half:
            offset -= period
        elif delta < -half:
            offset += period
        unwrapped.append(current + offset)
    return unwrapped


def principal_phase(value: complex) -> float:
    """复数传输值对应的主值相位（弧度）。"""
    return math.atan2(value.imag, value.real)
