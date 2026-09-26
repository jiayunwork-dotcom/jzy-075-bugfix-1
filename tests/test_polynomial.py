"""手写多项式求根（Durand-Kerner + Newton 精修）单元测试。"""

from __future__ import annotations

import cmath
import math

import pytest

from app.polynomial import (
    poly_from_roots,
    poly_multiply,
    roots_polynomial,
)


def test_poly_multiply_basic():
    assert poly_multiply([1, 2], [3, 4]) == pytest.approx([3, 10, 8])


def _assert_recovers_roots(coeffs_ascending: list[float], expected: list[complex]) -> None:
    roots = roots_polynomial(coeffs_ascending)
    assert len(roots) == len(expected)
    # 按到各期望根的最近距离配对
    remaining = list(expected)
    for root in roots:
        distances = [abs(root - target) for target in remaining]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        assert distances[nearest] == pytest.approx(0.0, abs=1e-7)
        remaining.pop(nearest)


def test_roots_real_distinct():
    # 根为 2、-3、0.5 的升幂多项式为 (1 - 2z^-1)(1 + 3z^-1)(1 - 0.5z^-1)
    coeffs = poly_multiply(poly_multiply([1, -2], [1, 3]), [1, -0.5])
    _assert_recovers_roots([float(c) for c in coeffs], [2 + 0j, -3 + 0j, 0.5 + 0j])


def test_roots_complex_conjugate():
    # 极点 0.9 e^{±j pi/3}
    r = 0.9
    angle = math.pi / 3
    pole = r * cmath.exp(1j * angle)
    coeffs = [1.0, -2 * r * math.cos(angle), r ** 2]
    _assert_recovers_roots(coeffs, [pole, pole.conjugate()])


def test_roots_from_iir_denominator():
    # 二阶节 (1 - 0.7 z^-1 + 0.1 z^-2)(1 + 0.5 z^-1)
    coeffs = poly_multiply([1, -0.7, 0.1], [1, 0.5])
    # 期望根：z^2 - 0.7 z + 0.1 的根为 0.5、0.2；外加 -0.5
    _assert_recovers_roots(
        [float(c) for c in coeffs], [0.5 + 0j, 0.2 + 0j, -0.5 + 0j]
    )


def test_roots_repeated_root():
    # (1 - 0.6 z^-1)^3
    coeffs = poly_multiply(poly_multiply([1, -0.6], [1, -0.6]), [1, -0.6])
    roots = roots_polynomial([float(c) for c in coeffs])
    assert len(roots) == 3
    for root in roots:
        assert root == pytest.approx(0.6 + 0j, abs=1e-7)


def test_roots_polynomial_with_trailing_zeros():
    # 高阶近零系数：0*z^-1... 升幂多项式末尾为 0 时实际降次
    # [1, -0.5, 0] 表示 1 - 0.5 z^-1，只有一个根 0.5
    roots = roots_polynomial([1.0, -0.5, 0.0])
    assert len(roots) == 1
    assert roots[0] == pytest.approx(0.5 + 0j, abs=1e-9)


def test_poly_from_roots_roundtrip():
    targets = [0.5 + 0j, -0.5 + 0.3j, -0.5 - 0.3j]
    coeffs = poly_from_roots(targets)
    roots = roots_polynomial([c.real for c in coeffs])
    for target in targets:
        assert min(abs(root - target) for root in roots) == pytest.approx(0.0, abs=1e-7)


def test_roots_invalid_inputs():
    with pytest.raises(ValueError):
        roots_polynomial([])
    with pytest.raises(ValueError):
        roots_polynomial([0.0, 0.0, 0.0])
