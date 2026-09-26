"""HTTP 接口层测试：三个接口的正常路径与非法输入拒绝。"""

from __future__ import annotations

import math

import pytest


def test_design_fir_hamming_returns_unified_format(client):
    response = client.post(
        "/api/v1/filters/design",
        json={"type": "fir", "order": 10, "cutoff": 0.3, "window": "hamming"},
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"b", "a"}
    assert len(data["b"]) == 11
    assert data["a"] == [1.0]
    assert all(isinstance(v, float) for v in data["b"] + data["a"])


def test_design_fir_accepts_hanning_alias(client):
    r1 = client.post(
        "/api/v1/filters/design",
        json={"type": "fir", "order": 6, "cutoff": 0.4, "window": "hann"},
    ).json()
    r2 = client.post(
        "/api/v1/filters/design",
        json={"type": "fir", "order": 6, "cutoff": 0.4, "window": "hanning"},
    ).json()
    assert r1["b"] == pytest.approx(r2["b"])


def test_design_iir_returns_unified_format(client):
    response = client.post(
        "/api/v1/filters/design",
        json={"type": "iir", "order": 4, "cutoff": 0.5},
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"b", "a"}
    assert len(data["b"]) == len(data["a"]) == 5
    assert data["a"][0] == pytest.approx(1.0)


def test_fir_requires_window(client):
    response = client.post(
        "/api/v1/filters/design",
        json={"type": "fir", "order": 4, "cutoff": 0.3},
    )
    assert response.status_code == 400
    assert "窗" in response.json()["error"]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "fir", "order": 0, "cutoff": 0.3, "window": "hamming"},
        {"type": "fir", "order": -2, "cutoff": 0.3, "window": "hamming"},
        {"type": "fir", "order": 17, "cutoff": 0.3, "window": "hamming"},
        {"type": "iir", "order": 100, "cutoff": 0.3},
        {"type": "fir", "order": 8, "cutoff": 0.0, "window": "hamming"},
        {"type": "fir", "order": 8, "cutoff": 1.0, "window": "hamming"},
        {"type": "iir", "order": 4, "cutoff": 1.0},
        {"type": "fir", "order": 8, "cutoff": 0.3, "window": "blackman"},
        {"type": "iir", "order": 4, "cutoff": 0.3, "window": "hann"},
    ],
)
def test_design_rejects_invalid_parameters(client, payload):
    response = client.post("/api/v1/filters/design", json=payload)
    assert response.status_code == 400, response.text
    assert response.json()["error"]


def test_design_unknown_filter_type_is_422(client):
    response = client.post(
        "/api/v1/filters/design",
        json={"type": "wavelet", "order": 4, "cutoff": 0.3},
    )
    assert response.status_code == 422


def test_design_order_true_rejected(client):
    # true 在 JSON 里会被解析成布尔，不能悄悄当成阶数 1
    response = client.post(
        "/api/v1/filters/design",
        json={"type": "iir", "order": True, "cutoff": 0.3},
    )
    assert response.status_code == 422


def test_end_to_end_design_then_analyze(client):
    designed = client.post(
        "/api/v1/filters/design",
        json={"type": "iir", "order": 3, "cutoff": 0.25},
    ).json()
    freq = client.post(
        "/api/v1/filters/frequency",
        json={
            "b": designed["b"],
            "a": designed["a"],
            "frequencies": [0.0, 0.25 * math.pi, math.pi],
        },
    )
    assert freq.status_code == 200
    points = freq.json()["points"]
    assert len(points) == 3
    assert points[0]["magnitude_db"] == pytest.approx(0.0, abs=1e-6)
    assert points[1]["magnitude_db"] == pytest.approx(-3.0103, abs=0.02)

    zp = client.post(
        "/api/v1/filters/zero-poles",
        json={"b": designed["b"], "a": designed["a"]},
    )
    assert zp.status_code == 200
    assert zp.json()["stable"] is True
    assert len(zp.json()["poles"]) == 3


def test_zero_poles_flags_unstable_via_http(client):
    response = client.post(
        "/api/v1/filters/zero-poles",
        json={"b": [1.0], "a": [1.0, -1.5]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["stable"] is False
    assert data["poles"][0]["distance_to_unit_circle"] == pytest.approx(0.5)
