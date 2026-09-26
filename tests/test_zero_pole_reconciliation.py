"""零极点对账回归测试：把设计接口出的系数走零极点接口，
再用增益/零点/极点在单位圆上重组传输函数，与频响接口直接代入
的结果逐点对账。这组用例钉住曾经出过的几类问题：

- FIR 求根飞离（零点模长几十、上亿）、复数零点丢共轭；
- 高阶巴特沃斯 N 重根 -1 合并不完整、散簇被误并、实轴投影丢根；
- 低截止高阶 IIR 贴轴共轭极点被误判成两个实根。

覆盖参数范围：FIR 三种窗、阶数 2..16；巴特沃斯阶数 1..16；
截止 0.2..0.8。对账频点从直流到截止频率均匀取 11 个。
"""

from __future__ import annotations

import math

import pytest

from app.analysis import (
    frequency_response,
    response_from_zero_poles,
    zero_pole_analysis,
)
from app.butterworth import design_iir_lowpass
from app.fir import design_fir_lowpass
from tests.conftest import linspace

WINDOWS = ("rectangular", "hann", "hamming")
FIR_ORDERS = tuple(range(2, 17))
BUTTER_ORDERS = tuple(range(1, 17))
CUTOFFS = (0.2, 0.35, 0.5, 0.65, 0.8)

# 对账容差与离群根模长上限（用户验收口径）
DB_TOLERANCE = 0.001
CONJUGATE_TOLERANCE = 1e-6
OUTLIER_RADIUS = 1.0e4


def _reconcile(b: list[float], a: list[float], cutoff: float) -> dict[str, float]:
    """返回对账指标：最大 dB 差、未配对复根个数、最大根模长。"""
    zp = zero_pole_analysis(b, a)
    zeros = [complex(z["real"], z["imag"]) for z in zp["zeros"]]
    poles = [complex(p["real"], p["imag"]) for p in zp["poles"]]

    def unpaired(roots: list[complex]) -> int:
        count = 0
        for root in roots:
            if abs(root.imag) <= 1e-9:
                continue
            nearest_mate = min(abs(root - other.conjugate()) for other in roots)
            if nearest_mate > CONJUGATE_TOLERANCE:
                count += 1
        return count

    frequencies = linspace(0.0, cutoff * math.pi, 11)
    db_coefficients = [
        point.magnitude_db
        for point in frequency_response(b, a, frequencies)
    ]
    db_zero_poles = response_from_zero_poles(
        zeros, poles, zp["gain"], frequencies
    )
    max_diff = max(abs(x - y) for x, y in zip(db_coefficients, db_zero_poles))
    max_radius = max(
        (abs(root) for root in zeros + poles),
        default=0.0,
    )
    return {
        "max_diff_db": max_diff,
        "unpaired_zeros": unpaired(zeros),
        "unpaired_poles": unpaired(poles),
        "max_radius": max_radius,
    }


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("order", FIR_ORDERS)
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_fir_zero_poles_reconcile_with_frequency_response(
    window, order, cutoff
):
    design = design_fir_lowpass(order, cutoff, window)
    metrics = _reconcile(design["b"], design["a"], cutoff)

    assert metrics["max_diff_db"] <= DB_TOLERANCE, (
        f"FIR {window} 阶数 {order} 截止 {cutoff}: "
        f"零极点重组与频响相差 {metrics['max_diff_db']:.4f} dB"
    )
    assert metrics["unpaired_zeros"] == 0
    assert metrics["unpaired_poles"] == 0
    assert metrics["max_radius"] < OUTLIER_RADIUS


@pytest.mark.parametrize("order", BUTTER_ORDERS)
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_butterworth_zero_poles_reconcile_with_frequency_response(
    order, cutoff
):
    design = design_iir_lowpass(order, cutoff)
    metrics = _reconcile(design["b"], design["a"], cutoff)

    assert metrics["max_diff_db"] <= DB_TOLERANCE, (
        f"巴特沃斯阶数 {order} 截止 {cutoff}: "
        f"零极点重组与频响相差 {metrics['max_diff_db']:.4f} dB"
    )
    assert metrics["unpaired_zeros"] == 0
    assert metrics["unpaired_poles"] == 0
    assert metrics["max_radius"] < OUTLIER_RADIUS


# ---------------------------------------------------------------------------
# 用户上报过的具体组合（定点钉死，防止数值改动悄悄回退）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order,cutoff,window,expected_dc_db",
    [
        (8, 0.5, "hann", 0.1063),       # 旧实现：零点飞到模长 ~186，直流高 33.6 dB
        (6, 0.5, "rectangular", None),  # 旧实现：零点模长上亿，直流高 ~155 dB
        (16, 0.3, "hamming", None),     # 旧实现：直流高约 23 dB
    ],
)
def test_reported_fir_cases_reconcile(order, cutoff, window, expected_dc_db):
    design = design_fir_lowpass(order, cutoff, window)
    zp = zero_pole_analysis(design["b"], design["a"])
    zeros = [complex(z["real"], z["imag"]) for z in zp["zeros"]]

    # 求根数量必须与非零系数降次后的阶数一致：前导零补成 z=0 根、
    # 末端零对应无穷远根直接降次（见 analysis._to_plane_roots）。
    leading_zeros = 0
    for coefficient in design["b"]:
        if coefficient == 0.0:
            leading_zeros += 1
        else:
            break
    trailing_zeros = 0
    for coefficient in reversed(design["b"]):
        if coefficient == 0.0:
            trailing_zeros += 1
        else:
            break
    assert len(zeros) == order - trailing_zeros
    for zero in zeros:
        assert math.isfinite(zero.real) and math.isfinite(zero.imag)
        assert abs(zero) < OUTLIER_RADIUS

    metrics = _reconcile(design["b"], design["a"], cutoff)
    assert metrics["max_diff_db"] <= DB_TOLERANCE
    assert metrics["unpaired_zeros"] == 0

    if expected_dc_db is not None:
        dc = frequency_response(design["b"], design["a"], [0.0])[0].magnitude_db
        assert dc == pytest.approx(expected_dc_db, abs=0.01)


@pytest.mark.parametrize("order", [7, 8, 9, 10, 12, 16])
def test_reported_butterworth_cases_reconcile(order):
    design = design_iir_lowpass(order, 0.5)
    metrics = _reconcile(design["b"], design["a"], 0.5)
    assert metrics["max_diff_db"] <= DB_TOLERANCE
    assert metrics["unpaired_zeros"] == 0
    # 高阶散开的根也必须是这组系数的根：对账通过即证明


def test_butterworth_order5_zeros_still_pinned_to_minus_one():
    """5 阶巴特沃斯零点贴 -1 的精度不能退化。"""
    design = design_iir_lowpass(5, 0.5)
    zp = zero_pole_analysis(design["b"], design["a"])
    assert len(zp["zeros"]) == 5
    for zero in zp["zeros"]:
        assert zero["real"] == pytest.approx(-1.0, abs=1e-9)
        assert zero["imag"] == pytest.approx(0.0, abs=1e-9)
        assert zero["on_unit_circle"] is True


def test_hann_endpoint_zero_still_yields_origin_zero():
    """汉宁窗首尾系数为零时，z=0 处那个零点照旧给出。"""
    design = design_fir_lowpass(10, 0.3, "hann")
    assert design["b"][0] == 0.0
    assert design["b"][-1] == 0.0
    zp = zero_pole_analysis(design["b"], design["a"])
    origin = [
        z
        for z in zp["zeros"]
        if abs(z["real"]) < 1e-12 and abs(z["imag"]) < 1e-12
    ]
    assert len(origin) == 1


def test_iir_pole_stability_metrics_preserved():
    """IIR 极点半径、到单位圆距离、不稳定口径保持不变。"""
    design = design_iir_lowpass(8, 0.3)
    zp = zero_pole_analysis(design["b"], design["a"])
    assert zp["stable"] is True
    assert len(zp["poles"]) == 8
    for pole in zp["poles"]:
        assert pole["radius"] < 1.0
        assert pole["distance_to_unit_circle"] == pytest.approx(
            pole["radius"] - 1.0, abs=1e-12
        )
        assert pole["distance_to_unit_circle"] < 0.0
        assert pole["on_unit_circle"] is False
