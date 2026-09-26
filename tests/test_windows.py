"""窗函数本身的测试（含 hanning/hann 同名归一化、奇偶数长度对称性）。"""

from __future__ import annotations

import math

import pytest

from app.errors import FilterError
from app.windows import get_window, hann, hamming, normalize_window_name, rectangular


@pytest.mark.parametrize("length", [1, 2, 7, 8, 17])
def test_windows_are_symmetric(length):
    for builder in (rectangular, hann, hamming):
        window = builder(length)
        assert len(window) == length
        for n in range(length):
            assert window[n] == pytest.approx(window[length - 1 - n], abs=1e-15)


def test_window_endpoint_values():
    assert rectangular(5) == [1.0] * 5
    assert hann(5)[0] == pytest.approx(0.0, abs=1e-15)
    assert hann(5)[2] == pytest.approx(1.0)
    assert hamming(5)[0] == pytest.approx(0.08, abs=1e-12)
    assert hamming(5)[2] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("alias", ["hann", "HANN", " hanning ", "Hanning", "汉宁窗"])
def test_hann_hanning_treated_as_same_window(alias):
    assert normalize_window_name(alias) == "hann"
    assert get_window(alias, 6) == pytest.approx(hann(6))


@pytest.mark.parametrize("name", ["blackman", "kaiser", "", "汉明", "hannn"])
def test_unknown_window_rejected(name):
    with pytest.raises(FilterError):
        normalize_window_name(name)


def test_window_length_must_be_positive():
    with pytest.raises(FilterError):
        get_window("hann", 0)
    with pytest.raises(FilterError):
        hamming(-3)
