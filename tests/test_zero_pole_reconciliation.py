"""零极点接口与频响接口的全参数对账（用户验收标准，端到端走 HTTP）。

对账流程（与用户手工流程一致）：
1. 设计接口出系数（系数本身不在本文件断言范围内，由其它测试锁定）；
2. 零极点接口拿增益、零点、极点；
3. 在单位圆上把 gain * prod(z - z_i) / prod(z - p_k) 乘回去，
   与频响接口在同一批频点的幅度 dB 对比。

判据：
- FIR 三种窗阶数 2..16、巴特沃斯阶数 1..16、截止 0.2..0.8，
  从直流到截止频率均匀取 11 个频点，两路偏差 <= 0.001 dB；
- 实系数下复零点共轭成对，配对偏差 <= 1e-6；
- 不出现离群零点（巴特沃斯零点必须贴在 z=-1 附近）；
- 同样的请求反复发，结果一模一样。
"""

from __future__ import annotations

import cmath
import math

import pytest

from tests.conftest import linspace

CUTOFFS = [0.2, 0.3, 0.5, 0.65, 0.8]
FIR_ORDERS = range(2, 17)
IIR_ORDERS = range(1, 17)
WINDOWS = ["rectangular", "hann", "hamming"]

RECONCILE_TOL_DB = 1e-3  # 零极点重组与频响接口的允许偏差
CONJUGATE_TOL = 1e-6  # 复零点共轭配对允许偏差
FIR_ZERO_RADIUS_LIMIT = 1e4  # FIR 零点模长上限（真根最大约 3e2，离群根曾到 1e8）


def _design(client, payload):
    response = client.post("/api/v1/filters/design", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    return data["b"], data["a"]


def _zero_poles(client, b, a):
    response = client.post("/api/v1/filters/zero-poles", json={"b": b, "a": a})
    assert response.status_code == 200, response.text
    return response.json()


def _frequency_db(client, b, a, freqs):
    response = client.post(
        "/api/v1/filters/frequency",
        json={"b": b, "a": a, "frequencies": freqs},
    )
    assert response.status_code == 200, response.text
    return [p["magnitude_db"] for p in response.json()["points"]]


def _reconstruct_db(zeros, poles, gain, w):
    """用户在单位圆上手工乘回去的幅度：gain * prod(z-z_i) / prod(z-p_k)。"""
    z = cmath.exp(1j * w)
    value = gain + 0j
    for zero in zeros:
        value *= z - zero
    for pole in poles:
        value /= z - pole
    return 20.0 * math.log10(abs(value))


def _reconciliation_error_db(client, b, a, cutoff):
    """直流到截止均匀 11 点，零极点重组与频响接口的最大幅度差（dB）。"""
    zp = _zero_poles(client, b, a)
    zeros = [complex(z["real"], z["imag"]) for z in zp["zeros"]]
    poles = [complex(p["real"], p["imag"]) for p in zp["poles"]]
    freqs = linspace(0.0, cutoff * math.pi, 11)
    reference = _frequency_db(client, b, a, freqs)
    rebuilt = [_reconstruct_db(zeros, poles, zp["gain"], w) for w in freqs]
    max_err = max(abs(x - y) for x, y in zip(reference, rebuilt, strict=True))
    return max_err, zp


def _assert_conjugate_paired(roots):
    """实系数多项式的复根必须共轭成对（配对偏差 <= 1e-6）。"""
    used = [False] * len(roots)
    for i, z in enumerate(roots):
        if used[i] or abs(z.imag) <= 1e-12:
            continue
        gap, partner = None, None
        for j, w in enumerate(roots):
            if j == i or used[j]:
                continue
            distance = abs(w - z.conjugate())
            if gap is None or distance < gap:
                gap, partner = distance, j
        assert partner is not None, f"根 {z} 找不到共轭伙伴"
        assert gap <= CONJUGATE_TOL, f"根 {z} 与共轭伙伴偏差 {gap:.2e} 超过 1e-6"
        used[partner] = True


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("order", FIR_ORDERS)
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_fir_zero_pole_reconciliation(client, window, order, cutoff):
    b, a = _design(
        client, {"type": "fir", "order": order, "cutoff": cutoff, "window": window}
    )
    err, zp = _reconciliation_error_db(client, b, a, cutoff)
    assert err <= RECONCILE_TOL_DB, (
        f"FIR {window} {order}阶 截止{cutoff}: 零极点重组与频响差 {err:.2e} dB"
    )
    zeros = [complex(z["real"], z["imag"]) for z in zp["zeros"]]
    _assert_conjugate_paired(zeros)
    # 不出现离群零点（真根最大约 3e2，且必带倒数伙伴；离群根曾到 1e8）
    assert max((abs(z) for z in zeros), default=0.0) <= FIR_ZERO_RADIUS_LIMIT


@pytest.mark.parametrize("order", IIR_ORDERS)
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_iir_zero_pole_reconciliation(client, order, cutoff):
    b, a = _design(client, {"type": "iir", "order": order, "cutoff": cutoff})
    err, zp = _reconciliation_error_db(client, b, a, cutoff)
    assert err <= RECONCILE_TOL_DB, (
        f"IIR {order}阶 截止{cutoff}: 零极点重组与频响差 {err:.2e} dB"
    )
    zeros = [complex(z["real"], z["imag"]) for z in zp["zeros"]]
    poles = [complex(p["real"], p["imag"]) for p in zp["poles"]]
    _assert_conjugate_paired(zeros)
    _assert_conjugate_paired(poles)
    # 巴特沃斯零点必须贴在 z=-1 附近（允许数值散开，不许离群到几十开外）
    assert len(zeros) == order
    for z in zeros:
        assert abs(z + 1) <= 0.5
    assert zp["stable"] is True


@pytest.mark.parametrize(
    "payload",
    [
        # 用户上报出现离群零点 / 对账差几十 dB 的参数组合，逐一钉死
        {"type": "fir", "order": 8, "cutoff": 0.5, "window": "hann"},
        {"type": "fir", "order": 6, "cutoff": 0.5, "window": "rectangular"},
        {"type": "fir", "order": 16, "cutoff": 0.3, "window": "hamming"},
        {"type": "iir", "order": 8, "cutoff": 0.5},
        {"type": "iir", "order": 12, "cutoff": 0.5},
        {"type": "iir", "order": 16, "cutoff": 0.5},
    ],
)
def test_reported_outlier_cases_stay_fixed(client, payload):
    b, a = _design(client, payload)
    err, zp = _reconciliation_error_db(client, b, a, payload["cutoff"])
    assert err <= RECONCILE_TOL_DB
    zeros = [complex(z["real"], z["imag"]) for z in zp["zeros"]]
    _assert_conjugate_paired(zeros)
    # 这几组系数的真零点模长都在 10 以内，100 的上限专为挡离群根
    assert max(abs(z) for z in zeros) <= 100.0


@pytest.mark.parametrize("order", IIR_ORDERS)
def test_butterworth_zeros_stay_at_minus_one(client, order):
    """巴特沃斯零点全部位于 z=-1：低阶精度不退，高阶也不许散开离群。"""
    b, a = _design(client, {"type": "iir", "order": order, "cutoff": 0.3})
    zp = _zero_poles(client, b, a)
    assert len(zp["zeros"]) == order
    for z in zp["zeros"]:
        assert z["real"] == pytest.approx(-1.0, abs=1e-6)
        assert z["imag"] == pytest.approx(0.0, abs=1e-6)
        assert z["on_unit_circle"] is True


def test_zero_pole_results_are_deterministic(client):
    """同样的请求反复发，零极点结果必须一模一样。"""
    b, a = _design(client, {"type": "iir", "order": 12, "cutoff": 0.5})
    assert _zero_poles(client, b, a) == _zero_poles(client, b, a)
    b, a = _design(
        client, {"type": "fir", "order": 15, "cutoff": 0.6, "window": "hann"}
    )
    assert _zero_poles(client, b, a) == _zero_poles(client, b, a)
