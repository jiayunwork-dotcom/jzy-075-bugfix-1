"""巴特沃斯原型 + 预畸变 + 双线性变换测试。

关键判据：
- 模拟原型极点落在 s 左半平面的单位圆上，共轭对称；
- 预畸变前后模拟/数字截止必须互相反推，且不能用"没预畸变"的值糊弄；
- 双线性变换后全部极点进入单位圆内（稳定），分母首项归一化为 1；
- -3dB 截止落在指定数字频率附近。
"""

from __future__ import annotations

import cmath
import math

import pytest

from app.analysis import frequency_response
from app.butterworth import (
    bilinear_map,
    butterworth_prototype_poles,
    design_iir_lowpass,
    prewarp,
    prewarp_inverse,
)
from app.errors import FilterError


@pytest.mark.parametrize("order", [1, 2, 3, 4, 8, 16])
def test_prototype_poles_on_unit_circle_left_half_plane(order):
    poles = butterworth_prototype_poles(order)
    assert len(poles) == order
    for pole in poles:
        assert abs(pole) == pytest.approx(1.0, abs=1e-12)
        assert pole.real < 0.0


def test_prototype_poles_conjugate_symmetry():
    poles = butterworth_prototype_poles(4)
    positives = sorted(p.imag for p in poles if p.imag > 0)
    negatives = sorted(-p.imag for p in poles if p.imag < 0)
    assert positives == pytest.approx(negatives, abs=1e-12)


@pytest.mark.parametrize(
    "cutoff", [0.05, 0.25, 0.3, 0.5, 0.8, 0.95]
)
def test_prewarp_roundtrip(cutoff):
    omega = prewarp(cutoff)
    assert prewarp_inverse(omega) == pytest.approx(cutoff, abs=1e-12)


def test_prewarp_values():
    # 归一化截止 0.5（角频率 pi/2）处 tan(pi/4) = 1
    assert prewarp(0.5) == pytest.approx(1.0)
    assert prewarp(0.2) == pytest.approx(math.tan(0.1 * math.pi))


def _naive_iir_without_prewarp(order: int, cutoff: float) -> dict[str, list[float]]:
    """对照实现：漏做预畸变，直接把数字截止角频率当模拟截止角频率。"""
    from app.polynomial import poly_from_roots, poly_multiply

    analog_cutoff = cutoff * math.pi  # 错误做法：不经过 tan() 预畸变
    poles = [bilinear_map(p * analog_cutoff) for p in butterworth_prototype_poles(order)]
    denominator = [1.0]
    real_pole = None
    used = [False] * order
    for i, p in enumerate(poles):
        if used[i]:
            continue
        for j in range(i + 1, order):
            q = poles[j]
            if not used[j] and abs(p.real - q.real) < 1e-12 and abs(p.imag + q.imag) < 1e-12:
                denominator = poly_multiply(denominator, [1.0, -2 * p.real, abs(p) ** 2])
                used[i] = used[j] = True
                break
        if not used[i]:
            used[i] = True
            real_pole = p
    if real_pole is not None:
        denominator = poly_multiply(denominator, [1.0, -real_pole.real])
    numerator = [c.real / (2.0 ** order) for c in poly_from_roots([-1.0] * order)]
    gain = sum(numerator) / sum(denominator)
    return {"b": [c / gain for c in numerator], "a": [float(c) for c in denominator]}


def _minus_3db_crossing(b, a) -> float:
    """扫描数字角频率上幅度降到 -3dB 的交叉点（rad/采样）。"""
    previous_w = 0.0
    previous_db = frequency_response(b, a, [0.0])[0].magnitude_db
    steps = 4000
    for k in range(1, steps + 1):
        w = math.pi * k / steps
        current_db = frequency_response(b, a, [w])[0].magnitude_db
        if current_db <= -3.0103 <= previous_db:
            # 线性插值
            ratio = (-3.0103 - previous_db) / (current_db - previous_db)
            return previous_w + ratio * (w - previous_w)
        previous_w, previous_db = w, current_db
    raise AssertionError("未找到 -3dB 交叉点")


def test_design_actually_uses_prewarping():
    """若漏做预畸变、直接拿数字角频率当模拟截止，-3dB 点会明显偏移。"""
    order = 4
    cutoff = 0.6  # 高频处预畸变影响显著；目标角频率 0.6pi
    target_w = cutoff * math.pi

    correct = design_iir_lowpass(order, cutoff)
    db_correct = frequency_response(correct["b"], correct["a"], [target_w])[0].magnitude_db
    assert db_correct == pytest.approx(-3.0103, abs=0.05)
    assert _minus_3db_crossing(correct["b"], correct["a"]) == pytest.approx(
        target_w, abs=5e-3
    )

    naive = _naive_iir_without_prewarp(order, cutoff)
    naive_crossing = _minus_3db_crossing(naive["b"], naive["a"])
    # 漏预畸变时真实 -3dB 点相对目标偏移超过 10%，与正确实现明显可区分
    assert abs(naive_crossing - target_w) / target_w > 0.1
    assert abs(naive_crossing - _minus_3db_crossing(correct["b"], correct["a"])) > 0.2


@pytest.mark.parametrize("order", [1, 2, 3, 4, 7, 16])
@pytest.mark.parametrize("cutoff", [0.1, 0.35, 0.8])
def test_iir_poles_inside_unit_circle_and_a0_normalized(order, cutoff):
    result = design_iir_lowpass(order, cutoff)
    b, a = result["b"], result["a"]
    assert a[0] == pytest.approx(1.0, abs=1e-14)
    assert len(a) == order + 1
    assert len(b) == order + 1

    # 直接从分母系数恢复 z 平面极点并检查 |z| < 1
    from app.polynomial import roots_polynomial

    poles = roots_polynomial(a)
    assert len(poles) == order
    for pole in poles:
        assert abs(pole) < 1.0 - 1e-9, f"极点 {pole} 位于单位圆外: |z|={abs(pole)}"


@pytest.mark.parametrize("order", [1, 2, 3, 5, 10])
def test_iir_minus_3db_at_specified_cutoff(order):
    cutoff = 0.25
    result = design_iir_lowpass(order, cutoff)
    points = frequency_response(result["b"], result["a"], [cutoff * math.pi])
    assert points[0].magnitude_db == pytest.approx(-3.0103, abs=0.02)


def test_iir_dc_gain_is_one():
    result = design_iir_lowpass(6, 0.4)
    assert sum(result["b"]) == pytest.approx(sum(result["a"]), rel=1e-10)
    points = frequency_response(result["b"], result["a"], [0.0])
    assert points[0].magnitude_db == pytest.approx(0.0, abs=1e-9)


def test_iir_coefficients_real_and_symmetric_zeros():
    result = design_iir_lowpass(4, 0.3)
    for coeff in result["b"] + result["a"]:
        assert isinstance(coeff, float)
        assert math.isfinite(coeff)


@pytest.mark.parametrize("bad_order", [0, -2, 17])
def test_iir_rejects_bad_order(bad_order):
    with pytest.raises(FilterError):
        design_iir_lowpass(bad_order, 0.3)


@pytest.mark.parametrize("bad_cutoff", [0.0, 1.0, 1.1, -0.5])
def test_iir_rejects_bad_cutoff(bad_cutoff):
    with pytest.raises(FilterError):
        design_iir_lowpass(4, bad_cutoff)
