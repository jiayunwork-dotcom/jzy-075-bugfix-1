"""手写多项式求根（Aberth-Ehrlich 同时迭代 + 重根验证合并）单元测试。"""

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


def test_roots_high_multiplicity_at_minus_one():
    # (1 + z^-1)^16：16 重根在双精度下散开不可分辨，应合并回 -1 且严格为实
    coeffs = poly_from_roots([-1.0] * 16)
    roots = roots_polynomial([c.real for c in coeffs])
    assert len(roots) == 16
    for root in roots:
        assert root == pytest.approx(-1.0 + 0j, abs=1e-9)


def test_roots_mixed_multiplicities():
    # (1+z^-1)^5 * (1-0.3z^-1) * (1-0.7z^-1)：重根与单根并存
    coeffs = poly_from_roots([-1.0] * 5)
    coeffs = poly_multiply(coeffs, [1.0, -0.3])
    coeffs = poly_multiply(coeffs, [1.0, -0.7])
    roots = roots_polynomial([c.real for c in coeffs])
    assert len(roots) == 7
    at_minus_one = [r for r in roots if abs(r + 1.0) < 1e-9]
    assert len(at_minus_one) == 5
    for target in (0.3, 0.7):
        assert min(abs(r - target) for r in roots) == pytest.approx(0.0, abs=1e-9)


def test_roots_conjugate_paired_for_real_coefficients():
    # 分离的复根对 + 近重根对 + 实根：输出必须共轭成对到 1e-9
    pair = 0.8 * cmath.exp(1j * 0.7)
    close_pair = [-1.0 + 0.02j, -1.0 - 0.02j]
    targets = [pair, pair.conjugate(), *close_pair, 0.5 + 0j, -0.25 + 0j]
    coeffs = poly_from_roots(targets)
    roots = roots_polynomial([c.real for c in coeffs])
    assert len(roots) == len(targets)
    remaining = list(roots)
    for root in roots:
        if abs(root.imag) < 1e-12:
            continue
        partner = min(remaining, key=lambda r: abs(r - root.conjugate()))
        assert abs(partner - root.conjugate()) == pytest.approx(0.0, abs=1e-9)
        remaining.remove(partner)


def test_roots_reciprocal_conjugate_quadruple():
    # 线性相位 FIR 的典型根结构：r e^{±jθ} 与 r^-1 e^{±jθ} 四元组
    angle = 0.9
    targets = [
        2.5 * cmath.exp(1j * angle),
        2.5 * cmath.exp(-1j * angle),
        0.4 * cmath.exp(1j * angle),
        0.4 * cmath.exp(-1j * angle),
    ]
    coeffs = poly_from_roots(targets)
    roots = roots_polynomial([c.real for c in coeffs])
    for target in targets:
        assert min(abs(r - target) for r in roots) == pytest.approx(0.0, abs=1e-9)


def test_roots_all_real_distinct():
    # 全部根为实根（共轭初值对的伪平衡情形）：2、-3、0.5、1.2、-0.4
    coeffs = [1.0]
    for root in (2.0, -3.0, 0.5, 1.2, -0.4):
        coeffs = poly_multiply(coeffs, [1.0, -root])
    roots = roots_polynomial(coeffs)
    assert len(roots) == 5
    for target in (2.0, -3.0, 0.5, 1.2, -0.4):
        assert min(abs(r - target) for r in roots) == pytest.approx(0.0, abs=1e-9)
