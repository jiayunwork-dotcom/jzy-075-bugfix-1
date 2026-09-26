"""频响与零极点分析测试，以及两条求值路径的一致性。

一致性判据（需求里最不放心的一条）：同一组系数，
- 频响接口在单位圆上直接代入分子/分母多项式；
- 零极点接口先求根，再用零极点积重组传输函数；
在截止频率附近一批频点上，两路 dB 与衰减趋势必须一致。
"""

from __future__ import annotations

import math

import pytest

from app.analysis import (
    frequency_response,
    response_from_zero_poles,
    zero_pole_analysis,
)
from app.errors import FilterError
from tests.conftest import fir_coeffs, iir_coeffs, linspace, linear_slope


def _response_db(client, b, a, freqs):
    response = client.post(
        "/api/v1/filters/frequency",
        json={"b": b, "a": a, "frequencies": freqs},
    )
    assert response.status_code == 200, response.text
    return [p["magnitude_db"] for p in response.json()["points"]]


@pytest.mark.parametrize("design", ["fir", "iir"])
def test_coefficient_and_root_paths_agree_near_cutoff(client, design):
    cutoff = 0.3  # 归一化截止（相对奈奎斯特）
    if design == "fir":
        b, a = fir_coeffs(client, 12, cutoff, window="hamming")
    else:
        b, a = iir_coeffs(client, 6, cutoff)

    freqs = linspace(0.6 * cutoff * math.pi, 1.4 * cutoff * math.pi, 41)

    # 路径一：系数直接代入
    db_coefficients = _response_db(client, b, a, freqs)

    # 路径二：零极点求根后重组
    zp = zero_pole_analysis(b, a)
    zeros = [complex(z["real"], z["imag"]) for z in zp["zeros"]]
    poles = [complex(p["real"], p["imag"]) for p in zp["poles"]]
    db_roots = response_from_zero_poles(zeros, poles, zp["gain"], freqs)

    for x, y in zip(db_coefficients, db_roots, strict=True):
        assert x == pytest.approx(y, abs=2e-6), (
            f"{design}: 频响路径 {x:.6f} dB 与零极点路径 {y:.6f} dB 不一致"
        )

    # 衰减趋势（dB 对频率的斜率序列）两路也必须方向一致
    slopes_a = [db_coefficients[i + 1] - db_coefficients[i] for i in range(len(freqs) - 1)]
    slopes_b = [db_roots[i + 1] - db_roots[i] for i in range(len(freqs) - 1)]
    for sa, sb in zip(slopes_a, slopes_b, strict=True):
        # 近零的平坦段不做符号断言；有明显变化时方向必须相同
        if abs(sa) > 1e-3 or abs(sb) > 1e-3:
            assert sa * sb >= 0.0


def test_phase_is_unwrapped_for_fir(client):
    # 用矩形窗保证直流相位为 0（hamming 直流求和为负，相位从 pi 起）
    b, a = fir_coeffs(client, 14, 0.3, window="rectangular")
    freqs = linspace(0.0, math.pi, 200)
    response = client.post(
        "/api/v1/filters/frequency",
        json={"b": b, "a": a, "frequencies": freqs},
    )
    phases = [p["phase_rad"] for p in response.json()["points"]]
    # 展开后的相位不应出现接近 2pi 的相邻跳变
    for prev, current in zip(phases, phases[1:]):
        assert abs(current - prev) < math.pi
    # 整体相位范围可以远超 (-pi, pi]，证明确实展开过
    assert max(phases) - min(phases) > math.pi
    # 斜率对应群延时 -7；只取通带段（截止在 0.3pi，前 50 点到 0.25pi）
    slope = linear_slope(freqs[:50], phases[:50])
    assert slope == pytest.approx(-7.0, rel=3e-3)


def test_zero_poles_stable_fir(client):
    b, a = fir_coeffs(client, 8, 0.3, window="hamming")
    result = zero_pole_analysis(b, a)
    assert result["stable"] is True
    # FIR 的零点都有限，没有分母极点（a=[1] 是常数多项式）
    assert result["poles"] == []
    for pole in result["zeros"]:
        assert pole["distance_to_unit_circle"] == pytest.approx(pole["radius"] - 1.0)


def test_hann_fir_with_zero_endpoints_analyzable(client):
    """汉宁窗首尾样本严格为 0，设计出来的首/末系数为 0 也要能正常分析。"""
    b, a = fir_coeffs(client, 10, 0.3, window="hann")
    assert b[0] == 0.0
    assert b[-1] == 0.0

    zp = client.post("/api/v1/filters/zero-poles", json={"b": b, "a": a})
    assert zp.status_code == 200, zp.text
    data = zp.json()
    assert data["stable"] is True
    # 首端零系数对应 z=0 处一个零点；末端零系数对应无穷远根，不出现在结果里
    origin_zeros = [z for z in data["zeros"] if abs(z["real"]) < 1e-12 and abs(z["imag"]) < 1e-12]
    assert len(origin_zeros) == 1

    # 频响接口同样可用
    fr = client.post(
        "/api/v1/filters/frequency",
        json={"b": b, "a": a, "frequencies": [0.0, 0.1 * math.pi, math.pi]},
    )
    assert fr.status_code == 200, fr.text
    points = fr.json()["points"]
    # 10 阶汉宁窗过渡带较宽、通带略有波动，这里只要求接近单位增益
    assert abs(points[0]["magnitude_db"]) < 0.5
    assert points[-1]["magnitude_db"] < -30.0


def test_zero_poles_stable_iir(client):
    b, a = iir_coeffs(client, 5, 0.3)
    result = zero_pole_analysis(b, a)
    assert result["stable"] is True
    assert len(result["poles"]) == 5
    for pole in result["poles"]:
        assert pole["radius"] < 1.0
        assert pole["distance_to_unit_circle"] < 0.0
        assert pole["on_unit_circle"] is False
    # IIR 巴特沃斯 N 个零点全在 z=-1（数值求根有少量误差）
    assert len(result["zeros"]) == 5
    for zero in result["zeros"]:
        assert zero["real"] == pytest.approx(-1.0, abs=1e-5)
        assert zero["imag"] == pytest.approx(0.0, abs=1e-5)
        assert zero["on_unit_circle"] is True


def test_unstable_pole_flagged():
    # 手工构造一个含单位圆外极点的分母 (1 - 1.2 z^-1)
    result = zero_pole_analysis([1.0], [1.0, -1.2])
    assert result["stable"] is False
    pole = result["poles"][0]
    assert pole["radius"] == pytest.approx(1.2)
    assert pole["distance_to_unit_circle"] == pytest.approx(0.2)


def test_pole_on_unit_circle_reported():
    result = zero_pole_analysis([1.0], [1.0, -1.0])
    assert result["poles"][0]["on_unit_circle"] is True
    # 恰在圆上不判为"圆外不稳定"
    assert result["stable"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"b": [], "a": [1.0], "frequencies": [0.1]},
        {"b": [1.0], "a": [], "frequencies": [0.1]},
        {"b": [1.0], "a": [1.0], "frequencies": []},
        {"b": [1.0, "x"], "a": [1.0], "frequencies": [0.1]},
        {"b": [1.0], "a": [1.0], "frequencies": [0.1, None]},
        {"b": [1.0], "a": [1.0], "frequencies": [1.5 * math.pi]},
        {"b": [1.0], "a": [1.0], "frequencies": [-0.1]},
        {"b": [1.0], "a": [0.0], "frequencies": [0.1]},
    ],
)
def test_frequency_endpoint_rejects_bad_inputs(client, payload):
    response = client.post("/api/v1/filters/frequency", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()


def test_zero_pole_endpoint_rejects_bad_coefficients(client):
    response = client.post(
        "/api/v1/filters/zero-poles", json={"b": [], "a": [1.0]}
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_magnitude_db_and_phase_units():
    # 直通系统 H=1：0 dB、相位 0
    points = frequency_response([1.0], [1.0], [0.0, math.pi / 2, math.pi])
    for point in points:
        assert point.magnitude_db == pytest.approx(0.0, abs=1e-12)
        assert point.phase_rad == pytest.approx(0.0, abs=1e-12)
