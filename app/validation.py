"""统一的输入校验。HTTP 层与核心模块共用同一套判定，错误一律抛 FilterError。"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .errors import FilterError

MAX_ORDER = 16


def validate_order(order: object, max_order: int = MAX_ORDER) -> int:
    """阶数：必须是正整数（bool 不算），且不超过 max_order（默认 16）。"""
    if isinstance(order, bool) or not isinstance(order, int):
        if isinstance(order, float) and order.is_integer():
            order = int(order)
        else:
            raise FilterError("阶数必须是正整数")
    if order <= 0:
        raise FilterError(f"阶数必须为正整数，收到 {order}")
    if order > max_order:
        raise FilterError(f"阶数 {order} 超过上限 {max_order} 阶")
    return order


def validate_cutoff(cutoff: object) -> float:
    """归一化截止频率，相对奈奎斯特，必须严格落在开区间 (0, 1)。

    1.0 对应奈奎斯特频率（数字角频率 pi rad/采样）。
    """
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise FilterError("截止频率必须是数值")
    value = float(cutoff)
    if not math.isfinite(value):
        raise FilterError("截止频率必须是有限数值")
    if value <= 0.0:
        raise FilterError("截止频率不能为 0 或负数（必须严格大于 0）")
    if value >= 1.0:
        raise FilterError("截止频率不能达到或超过奈奎斯特频率（归一化值 1.0）")
    return value


def validate_coefficients(coeffs: object, name: str) -> list[float]:
    """校验分子 / 分母系数：非空、元素都是有限数值。

    分子首项允许为 0（汉宁窗 FIR 首尾系数严格为 0，等价于 z=0 处有零点，
    由 analysis 模块补根处理）；分母首项必须非零，否则传输函数无定义。
    """
    if not isinstance(coeffs, Sequence) or isinstance(coeffs, (str, bytes)):
        raise FilterError(f"{name} 必须是数组")
    if len(coeffs) == 0:
        raise FilterError(f"{name} 不能为空")
    cleaned: list[float] = []
    for index, item in enumerate(coeffs):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise FilterError(f"{name}[{index}] 不是数值")
        value = float(item)
        if not math.isfinite(value):
            raise FilterError(f"{name}[{index}] 必须是有限数值")
        cleaned.append(value)
    if not any(abs(value) > 0.0 for value in cleaned):
        raise FilterError(f"{name} 不能是全零多项式")
    if name.startswith("分母") and abs(cleaned[0]) < 1e-300:
        raise FilterError(f"{name} 首项不能为 0")
    return cleaned


def validate_frequencies(frequencies: object) -> list[float]:
    """校验频点数组：非空、元素全是有限数值、且都在 [0, pi] 内。"""
    if not isinstance(frequencies, Sequence) or isinstance(frequencies, (str, bytes)):
        raise FilterError("频率点必须是数组")
    if len(frequencies) == 0:
        raise FilterError("频率点数组不能为空")
    cleaned: list[float] = []
    for index, item in enumerate(frequencies):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise FilterError(f"频率点[{index}] 不是数值")
        value = float(item)
        if not math.isfinite(value):
            raise FilterError(f"频率点[{index}] 必须是有限数值")
        if not (0.0 <= value <= math.pi):
            raise FilterError(
                f"频率点[{index}]={value} 越界，归一化频率必须在 [0, pi] 内"
            )
        cleaned.append(value)
    return cleaned
