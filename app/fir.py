"""FIR 低通滤波器：窗函数法（核心过程手写，不调用黑盒设计函数）。

约定：``cutoff`` 是相对奈奎斯特的归一化截止频率，取值 (0, 1)，
对应数字角频率 wc = cutoff * pi (rad/采样)。

步骤：
1. 理想低通的冲激响应
       h_d[n] = sin(wc (n - alpha)) / (pi (n - alpha)),  alpha = order / 2
   中心点 n == alpha 处取极限值 wc / pi = cutoff。
2. 乘上同长度的对称窗 w[n]：b[n] = h_d[n] * w[n]。

阶数 order 对应 order + 1 个抽头，群延时恒为 order / 2 个采样，
系数关于 alpha 首尾对称（线性相位）。
"""

from __future__ import annotations

import math

from .errors import FilterError
from .validation import validate_cutoff, validate_order
from .windows import get_window


def ideal_lowpass_impulse(cutoff: float, order: int) -> list[float]:
    """理想（加窗前）低通冲激响应，长度 order + 1。

    cutoff 为相对奈奎斯特的归一化截止 (0,1)，内部换算成角频率 cutoff*pi。
    """
    omega_c = cutoff * math.pi
    alpha = order / 2.0
    h: list[float] = []
    for n in range(order + 1):
        shift = n - alpha
        if abs(shift) < 1e-14:
            # sin(x)/x 在 x=0 处的极限，即 wc/pi = cutoff
            h.append(cutoff)
        else:
            x = omega_c * shift
            h.append(math.sin(x) / (math.pi * shift))
    return h


def design_fir_lowpass(order: int, cutoff: float, window_name: str) -> dict[str, list[float]]:
    """窗函数法 FIR 低通。

    参数
    ----
    order: 滤波器阶数，抽头数为 order + 1。
    cutoff: 归一化截止角频率（rad/采样），相对奈奎斯特，即 0 < cutoff < pi。
    window_name: 窗类型（rectangular / hann / hanning / hamming）。
    """
    if not isinstance(window_name, str) or not window_name.strip():
        raise FilterError("FIR 设计必须指定窗类型")

    order = validate_order(order)
    cutoff = validate_cutoff(cutoff)
    taps = order + 1
    window = get_window(window_name, taps)  # 内部会校验窗名与长度
    ideal = ideal_lowpass_impulse(cutoff, order)

    coefficients = [h * w for h, w in zip(ideal, window, strict=True)]
    return {"b": coefficients, "a": [1.0]}
