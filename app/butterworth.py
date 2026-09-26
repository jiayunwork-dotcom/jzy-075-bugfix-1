"""巴特沃斯模拟原型 + 频率预畸变 + 双线性变换（核心逻辑全部手写）。

约定
----
``cutoff`` 是相对奈奎斯特的归一化截止频率，取值 (0, 1)，
对应数字角频率 w = cutoff * pi (rad/采样)。流程：

1. 在模拟域求巴特沃斯原型极点（3dB 截止 = 1 rad/s 的归一化原型，
   左半平面、关于实轴对称）；
2. 用双线性变换对数字截止做频率预畸变，得到模拟截止角频率 Omega_c，
   再把原型极点整体乘以 Omega_c 完成频率缩放；
3. 双线性变换（取 T = 2，公式 z = (1 + sT/2)/(1 - sT/2)，
   简化后 z = (1 + s)/(1 - s)），逐极点映到 z 平面；
4. 把分母按共轭极点对（奇数阶补一个实极点）展成实系数多项式，
   分子为 (1 + z^-1)^N / 2^N（N 个零点固定在 z = -1），
   最后整体缩放使直流增益为 1，即分母首项 a[0] 归一化为 1。
"""

from __future__ import annotations

import cmath
import math

from .errors import FilterError
from .polynomial import poly_from_roots, poly_multiply
from .validation import validate_cutoff, validate_order


def butterworth_prototype_poles(order: int) -> list[complex]:
    """归一化巴特沃斯低通原型在 s 平面左半平面的 ``order`` 个极点。

    p_k = exp(j * (pi/2 + pi(2k+1)/(2N))), k = 0..N-1，全部满足 |p_k| = 1。
    """
    if order <= 0:
        raise FilterError("阶数必须为正整数")
    poles: list[complex] = []
    for k in range(order):
        angle = math.pi / 2.0 + math.pi * (2 * k + 1) / (2.0 * order)
        poles.append(cmath.exp(1j * angle))
    return poles


def prewarp(cutoff: float) -> float:
    """归一化数字截止（相对奈奎斯特，(0,1)）-> 预畸变后的模拟截止。

    数字角频率 w = cutoff*pi；取 T = 2 时
    Omega = (2/T) tan(w/2) 简化为 Omega = tan(w/2) = tan(cutoff*pi/2)。
    """
    if not (0.0 < cutoff < 1.0):
        raise FilterError("截止频率必须落在开区间 (0, 1)（相对奈奎斯特）")
    return math.tan(cutoff * math.pi / 2.0)


def prewarp_inverse(omega: float) -> float:
    """预畸变的反函数：归一化截止 = 2 atan(Omega) / pi，供互相反推测试使用。"""
    if omega <= 0.0:
        raise FilterError("模拟截止角频率必须为正数")
    return 2.0 * math.atan(omega) / math.pi


def bilinear_map(pole_s: complex) -> complex:
    """单个模拟极点 s -> 数字极点 z（T = 2）：z = (1 + s)/(1 - s)。"""
    return (1.0 + pole_s) / (1.0 - pole_s)


def design_iir_lowpass(order: int, cutoff: float) -> dict[str, list[float]]:
    """巴特沃斯低通 + 双线性变换，返回统一格式的系数 {"b": [...], "a": [...]}。"""
    # 1. 模拟原型极点
    order = validate_order(order)
    cutoff = validate_cutoff(cutoff)
    prototype = butterworth_prototype_poles(order)

    # 2. 预畸变 + 频率缩放（必须先预畸变，不能拿 cutoff 直接当模拟截止）
    omega_c = prewarp(cutoff)
    analog_poles = [p * omega_c for p in prototype]

    # 3. 双线性变换到数字域
    digital_poles = [bilinear_map(p) for p in analog_poles]

    # 4. 按共轭对（奇阶补一个实极点）构造分母二阶节并累乘成多项式
    sections: list[list[float]] = []
    used = [False] * order
    real_pole: complex | None = None
    for i, p in enumerate(digital_poles):
        if used[i]:
            continue
        for j in range(i + 1, order):
            q = digital_poles[j]
            if (
                not used[j]
                and abs(p.real - q.real) < 1e-12
                and abs(p.imag + q.imag) < 1e-12
                and abs(p.imag) > 1e-12
            ):
                # (1 - p z^-1)(1 - conj(p) z^-1)
                sections.append([1.0, -2.0 * p.real, abs(p) ** 2])
                used[i] = used[j] = True
                break
        if not used[i]:
            real_pole = p
            used[i] = True
    if real_pole is not None:
        sections.append([1.0, -real_pole.real])

    denominator = [1.0]
    for section in sections:
        denominator = poly_multiply(denominator, section)

    # 分子：N 个零点全部位于 z = -1，先按未归一化增益 1/2^N 构造
    numerator = poly_from_roots([-1.0] * order)
    numerator = [coef / (2.0 ** order) for coef in numerator]

    # 5. 直流增益归一化到 1：直流处 H(1) = sum(b)/sum(a)，
    #    巴特沃斯直流增益本应为 1，缩放后分母首项 a[0] 正好归一化为 1。
    gain = sum(numerator) / sum(denominator)
    numerator = [coef / gain for coef in numerator]

    # 抹掉浮点噪声引入的极小虚部
    numerator = [float(c.real) if isinstance(c, complex) else float(c) for c in numerator]
    denominator = [
        float(c.real) if isinstance(c, complex) else float(c) for c in denominator
    ]

    if abs(denominator[0]) < 1e-14:
        raise FilterError("IIR 分母首项异常，无法归一化")
    return {"b": numerator, "a": denominator}
