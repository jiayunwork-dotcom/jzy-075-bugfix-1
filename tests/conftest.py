"""测试公共夹具与辅助函数。"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def fir_coeffs(client: TestClient, order: int, cutoff: float, window: str = "hamming"):
    response = client.post(
        "/api/v1/filters/design",
        json={"type": "fir", "order": order, "cutoff": cutoff, "window": window},
    )
    assert response.status_code == 200, response.text
    return response.json()["b"], response.json()["a"]


def iir_coeffs(client: TestClient, order: int, cutoff: float):
    response = client.post(
        "/api/v1/filters/design",
        json={"type": "iir", "order": order, "cutoff": cutoff},
    )
    assert response.status_code == 200, response.text
    return response.json()["b"], response.json()["a"]


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * k for k in range(count)]


def linear_slope(xs: list[float], ys: list[float]) -> float:
    """最小二乘斜率，用于检查相位的线性度。"""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return numerator / denominator
