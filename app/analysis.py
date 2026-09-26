"""频响求值与零极点分析。

两条分析路径刻意各自独立、但共用 polynomial 模块的同一份底层实现：
- 频响：直接把 z = e^{jw} 代入分子 / 分母多项式（Horner 求值）；
- 零极点：对分子 / 分母多项式求根，再用
  H(z) = 零点积 (z - z_i) / 极点积 (z - p_k) 重组频率响应。

自动化测试据此比较"系数代入"和"零极点重组"两路结果是否一致。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from .phase import principal_phase, unwrap_phase
from .polynomial import roots_polynomial
from .validation import validate_coefficients, validate_frequencies

_MIN_DB = -1000.0  # 传输值数值为 0 时的兜底下限


@dataclass(frozen=True)
class FrequencyPoint:
    frequency: float
    magnitude_db: float
    phase_rad: float


def _eval_ascending(coeffs: list[float], z: complex) -> complex:
    """Horner 法求 c0 + c1 z^-1 + ... + cN z^-N 在给定 z 处的值。"""
    value = 0j
    for coef in reversed(coeffs):
        value = value / z + coef
    return value


def _transfer_from_coefficients(
    b: list[float], a: list[float], z: complex
) -> complex:
    """H(z) = B(z^-1) / A(z^-1)，系数按 z^-1 升幂排列。"""
    numerator = _eval_ascending(b, z)
    denominator = _eval_ascending(a, z)
    if abs(denominator) < 1e-300:
        raise ZeroDivisionError("分母多项式在该频点数值为 0")
    return numerator / denominator


def _transfer_from_poles_zeros(
    zeros: list[complex], poles: list[complex], gain: float, z: complex
) -> complex:
    """由零极点重组：H(z) = gain * prod(z - z_i) / prod(z - p_k)。"""
    numerator = gain + 0j
    for zero in zeros:
        numerator *= z - zero
    denominator = 1.0 + 0j
    for pole in poles:
        denominator *= z - pole
    if abs(denominator) < 1e-300:
        raise ZeroDivisionError("极点积在该频点数值为 0")
    return numerator / denominator


def frequency_response(
    b: list[float], a: list[float], frequencies: list[float]
) -> list[FrequencyPoint]:
    """在单位圆 z = e^{jw} 上逐点求幅度(dB)与展开后相位(弧度)。"""
    b = validate_coefficients(b, "分子系数 b")
    a = validate_coefficients(a, "分母系数 a")
    freqs = validate_frequencies(frequencies)

    raw_phases: list[float] = []
    magnitudes: list[float] = []
    for w in freqs:
        z = cmath.exp(1j * w)
        h_value = _transfer_from_coefficients(b, a, z)
        amplitude = abs(h_value)
        if amplitude <= 1e-300:
            magnitudes.append(_MIN_DB)
        else:
            magnitudes.append(20.0 * math.log10(amplitude))
        raw_phases.append(principal_phase(h_value))

    phases = unwrap_phase(raw_phases)
    return [
        FrequencyPoint(frequency=w, magnitude_db=mag, phase_rad=phase)
        for w, mag, phase in zip(freqs, magnitudes, phases)
    ]


def _to_plane_roots(coeffs_ascending: list[float]) -> tuple[list[complex], float]:
    """升幂 z^-1 系数 -> (z 平面根, 首项非零系数)。

    常规情形 c0 != 0：c0 + c1 z^-1 + ... + cN z^-N 乘 z^N 后
    roots_polynomial 直接解出 z_i，增益取 c0。

    前导零情形（如汉宁窗 FIR 首系数严格为 0）：
    c0=...=c_{k-1}=0 时多项式等价于 z^-k 乘一个首项非零的多项式，
    z=0 处补 k 个根，增益取第一个非零系数。末端零系数对应 z=无穷大
    处的根，不影响有限频响，直接剥离。
    """
    leading_zeros = 0
    for coefficient in coeffs_ascending:
        if abs(coefficient) < 1e-15:
            leading_zeros += 1
        else:
            break

    trailing = len(coeffs_ascending)
    while trailing > 1 and abs(coeffs_ascending[trailing - 1]) < 1e-15:
        trailing -= 1
    trimmed = coeffs_ascending[leading_zeros:trailing]
    if not trimmed:
        raise ValueError("全零多项式无法求根")

    roots = roots_polynomial(trimmed)
    roots.extend([0j] * leading_zeros)
    roots.sort(key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    return roots, trimmed[0]


def zero_pole_analysis(b: list[float], a: list[float]) -> dict[str, object]:
    """求分子分母的根（零点 / 极点），报告到单位圆距离并标注稳定性。

    极点只要有一个严格位于单位圆外，系统即判为不稳定；位于单位圆上
    （如纯振荡的边界情形）单独标注 on_unit_circle，不算稳定极点。
    """
    b = validate_coefficients(b, "分子系数 b")
    a = validate_coefficients(a, "分母系数 a")

    zeros, b_top = _to_plane_roots(b)
    poles, a_top = _to_plane_roots(a)
    gain = b_top / a_top

    tolerance = 1e-9

    def describe(root: complex) -> dict[str, object]:
        radius = abs(root)
        on_circle = abs(radius - 1.0) < tolerance
        return {
            "real": root.real,
            "imag": root.imag,
            "radius": radius,
            "distance_to_unit_circle": radius - 1.0,
            "on_unit_circle": on_circle,
        }

    pole_items = [describe(p) for p in poles]
    zero_items = [describe(z_item) for z_item in zeros]
    unstable = any(item["distance_to_unit_circle"] > tolerance for item in pole_items)

    return {
        "zeros": zero_items,
        "poles": pole_items,
        "gain": gain,
        "stable": not unstable,
    }


def response_from_zero_poles(
    zeros: list[complex],
    poles: list[complex],
    gain: float,
    frequencies: list[float],
) -> list[float]:
    """零极点路径下的单位圆幅度(dB)，供一致性测试使用。"""
    freqs = validate_frequencies(frequencies)
    result: list[float] = []
    for w in freqs:
        z = cmath.exp(1j * w)
        h_value = _transfer_from_poles_zeros(zeros, poles, gain, z)
        amplitude = abs(h_value)
        result.append(
            _MIN_DB if amplitude <= 1e-300 else 20.0 * math.log10(amplitude)
        )
    return result
