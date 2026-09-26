"""HTTP 请求 / 响应的 JSON 模型。

阶数、截止频率这类业务规则由 validation 模块在路由内显式校验，
模型层只负责 JSON 类型与布尔伪装（true 会被 Pydantic 当 int）的拦截。
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

FilterType = Literal["fir", "iir"]


def _reject_bool(value: object) -> object:
    """JSON 里的 true/false 不能冒充整数 / 浮点。"""
    if isinstance(value, bool):
        raise ValueError("不能使用布尔值 true/false，需要数值")
    return value


StrictInt = Annotated[int, BeforeValidator(_reject_bool)]
StrictFloat = Annotated[float, BeforeValidator(_reject_bool)]


class DesignRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: FilterType = Field(..., description="fir 或 iir")
    order: StrictInt = Field(..., description="滤波器阶数，1..16")
    cutoff: StrictFloat = Field(..., description="归一化截止频率（相对奈奎斯特），0 < cutoff < 1")
    window: str | None = Field(None, description="FIR 必给：rectangular/hann/hanning/hamming")

    @field_validator("cutoff")
    @classmethod
    def _cutoff_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("截止频率必须是有限数值")
        return value


class Coefficients(BaseModel):
    model_config = ConfigDict(extra="ignore")

    b: list[float]
    a: list[float]


class CoefficientsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # 元素类型刻意不在这里限定：交给 validation 模块统一拒绝并给出中文原因
    b: list
    a: list


class FrequencyResponseRequest(CoefficientsRequest):
    frequencies: list


class FrequencyPointOut(BaseModel):
    frequency: float
    magnitude_db: float
    phase_rad: float


class FrequencyResponseResponse(BaseModel):
    points: list[FrequencyPointOut]


class RootItem(BaseModel):
    real: float
    imag: float
    radius: float
    distance_to_unit_circle: float
    on_unit_circle: bool


class ZeroPoleResponse(BaseModel):
    zeros: list[RootItem]
    poles: list[RootItem]
    gain: float
    stable: bool
