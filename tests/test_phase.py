"""手写相位展开的单元测试。"""

from __future__ import annotations

import math

import pytest

from app.phase import principal_phase, unwrap_phase


def test_unwrap_no_jump_inside_branch():
    phases = [0.0, 0.5, 1.0, 1.5]
    assert unwrap_phase(phases) == phases


def test_unwrap_positive_branch_crossing():
    # 3.0 -> -3.0 主值上跳变表面为 -6，实际连续变化应补 +2pi
    phases = [3.0, -3.0, -2.5]
    result = unwrap_phase(phases)
    assert result[0] == 3.0
    assert result[1] == pytest.approx(-3.0 + 2 * math.pi)
    assert result[2] == pytest.approx(-2.5 + 2 * math.pi)


def test_unwrap_negative_branch_crossing():
    phases = [-3.0, 3.0, 2.5]
    result = unwrap_phase(phases)
    assert result[1] == pytest.approx(3.0 - 2 * math.pi)
    assert result[2] == pytest.approx(2.5 - 2 * math.pi)


def test_unwrap_multiple_windings():
    # 线性斜率扫过多个 2pi：连续相位 = -2*w*order 之类
    order = 10
    steps = 400
    continuous = [-2.0 * math.pi * order * k / steps for k in range(steps + 1)]
    wrapped = [math.atan2(math.sin(p), math.cos(p)) for p in continuous]
    result = unwrap_phase(wrapped)
    for got, expected in zip(result, continuous, strict=True):
        assert got == pytest.approx(expected, abs=1e-9)


def test_principal_phase_range():
    for value in (2 + 0j, -2 + 0j, 0 + 1j, 0 - 1j, -1 - 1j):
        angle = principal_phase(value)
        assert -math.pi < angle <= math.pi


def test_unwrap_empty():
    assert unwrap_phase([]) == []
