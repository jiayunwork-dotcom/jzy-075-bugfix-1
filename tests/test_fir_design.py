"""FIR 窗函数法测试：
- 系数首尾对称（线性相位），容差内验证；
- 阶数奇/偶切换时抽头数跟着变、尾巴不截歪；
- 通带内相位斜率与阶数换算的群延时一致。
"""

from __future__ import annotations

import math

import pytest

from app.analysis import frequency_response
from app.errors import FilterError
from app.fir import design_fir_lowpass
from app.validation import validate_order
from tests.conftest import linear_slope, linspace

CUTOFF = 0.3  # 相对奈奎斯特的归一化截止频率
CUTOFF_W = CUTOFF * math.pi  # 对应数字角频率
SYMMETRY_TOL = 1e-12


def assert_coefficients_symmetric(b: list[float], tol: float = SYMMETRY_TOL) -> None:
    n = len(b)
    for k in range(n // 2):
        assert b[k] == pytest.approx(b[n - 1 - k], abs=tol), (
            f"系数不对称: b[{k}]={b[k]} vs b[{n - 1 - k}]={b[n - 1 - k]}"
        )


@pytest.mark.parametrize("order", [1, 2, 3, 4, 8, 15, 16])
@pytest.mark.parametrize("window", ["rectangular", "hann", "hamming"])
def test_fir_coefficients_are_symmetric(order, window):
    result = design_fir_lowpass(order, CUTOFF, window)
    b, a = result["b"], result["a"]
    assert a == [1.0]
    assert len(b) == order + 1
    assert_coefficients_symmetric(b)


@pytest.mark.parametrize("window", ["hann", "hanning"])
def test_hann_aliases_produce_identical_coefficients(window):
    b_hann = design_fir_lowpass(10, CUTOFF, "hann")["b"]
    b_hanning = design_fir_lowpass(10, CUTOFF, "hanning")["b"]
    assert b_hann == pytest.approx(b_hanning)


def test_odd_even_order_changes_tap_count_and_keeps_balance():
    """奇数阶(偶数抽头)与偶数阶(奇数抽头)切换时长度与对称中心一起变。"""
    b_even_order = design_fir_lowpass(8, CUTOFF, "hamming")["b"]  # 9 抽头
    b_odd_order = design_fir_lowpass(7, CUTOFF, "hamming")["b"]   # 8 抽头
    assert len(b_even_order) == 9
    assert len(b_odd_order) == 8
    assert_coefficients_symmetric(b_even_order)
    assert_coefficients_symmetric(b_odd_order)
    # 偶数抽头时对称中心落在两个样本中间（半采样群延时 3.5），不能截出歪尾巴
    assert sum(b_odd_order) == pytest.approx(2 * sum(b_odd_order[:4]), rel=1e-12)


def test_fir_phase_linearity_matches_group_delay():
    """通带内展开相位斜率 -group_delay 应与 order/2 对得上。

    用矩形窗并在明显处于通带的区间（0.005pi..0.18pi，远低于截止 0.3pi）
    密采样，保证测得的是通带线性相位而非过渡带的相位弯曲。
    """
    order = 12
    result = design_fir_lowpass(order, CUTOFF, "rectangular")
    freqs = linspace(0.005 * math.pi, 0.18 * math.pi, 40)
    points = frequency_response(result["b"], result["a"], freqs)
    xs = [p.frequency for p in points]
    ys = [p.phase_rad for p in points]
    slope = linear_slope(xs, ys)
    expected_delay = order / 2.0
    assert slope == pytest.approx(-expected_delay, rel=1e-3, abs=1e-3)


def test_fir_dc_gain_near_one():
    result = design_fir_lowpass(12, CUTOFF, "hamming")
    points = frequency_response(result["b"], result["a"], [0.0])
    assert points[0].magnitude_db == pytest.approx(0.0, abs=0.02)


def test_fir_stopband_attenuates():
    result = design_fir_lowpass(12, CUTOFF, "hamming")
    points = frequency_response(result["b"], result["a"], [0.9 * math.pi])
    assert points[0].magnitude_db < -20.0

@pytest.mark.parametrize("bad_order", [0, -1, -16, 17, 100])
def test_fir_rejects_bad_order(bad_order):
    with pytest.raises(FilterError):
        validate_order(bad_order)
    with pytest.raises(FilterError):
        design_fir_lowpass(bad_order, CUTOFF, "hamming")


@pytest.mark.parametrize("bad_cutoff", [0.0, -0.1, 1.0, 1.5, float("nan")])
def test_fir_rejects_bad_cutoff(bad_cutoff):
    with pytest.raises(FilterError):
        design_fir_lowpass(8, bad_cutoff, "hamming")


def test_fir_rejects_unknown_window():
    with pytest.raises(FilterError):
        design_fir_lowpass(8, CUTOFF, "blackman")


def test_order_upper_bound_is_sixteen():
    assert validate_order(16) == 16
    with pytest.raises(FilterError, match="上限"):
        validate_order(17)
