"""FastAPI 路由层：三个 HTTP 接口，传 JSON 拿 JSON。

POST /api/v1/filters/design        设计 FIR（窗函数法）或 IIR（巴特沃斯+双线性）
POST /api/v1/filters/frequency     频响：幅度 dB + 展开相位（弧度）
POST /api/v1/filters/zero-poles    零极点：到单位圆距离 + 不稳定标注
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .analysis import frequency_response, zero_pole_analysis
from .butterworth import design_iir_lowpass
from .errors import FilterError
from .fir import design_fir_lowpass
from .schemas import (
    CoefficientsRequest,
    DesignRequest,
    FrequencyResponseRequest,
    FrequencyResponseResponse,
    ZeroPoleResponse,
)
from .validation import validate_cutoff, validate_order
from .windows import normalize_window_name

app = FastAPI(
    title="低通滤波器设计与分析服务",
    version=__version__,
    description="FIR 窗函数法 / IIR 巴特沃斯双线性变换设计，频响与零极点分析。",
)


@app.exception_handler(FilterError)
async def filter_error_handler(_request: Request, exc: FilterError) -> JSONResponse:
    """业务错误统一成 400，并带上中文原因。"""
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/api/v1/filters/design")
async def design_filter(request: DesignRequest) -> dict[str, list[float]]:
    order = validate_order(request.order)
    cutoff = validate_cutoff(request.cutoff)

    if request.type == "fir":
        if request.window is None:
            raise FilterError("FIR 设计必须通过 window 字段指定窗类型")
        window_name = normalize_window_name(request.window)
        return design_fir_lowpass(order, cutoff, window_name)

    # IIR 不需要窗；调用方误传时明确拒绝，避免参数被静默忽略
    if request.type == "iir":
        if request.window is not None:
            raise FilterError("IIR（巴特沃斯）设计不接受 window 参数")
        return design_iir_lowpass(order, cutoff)


@app.post(
    "/api/v1/filters/frequency",
    response_model=FrequencyResponseResponse,
)
async def evaluate_frequency(
    request: FrequencyResponseRequest,
) -> FrequencyResponseResponse:
    points = frequency_response(request.b, request.a, request.frequencies)
    return FrequencyResponseResponse(
        points=[
            {
                "frequency": point.frequency,
                "magnitude_db": point.magnitude_db,
                "phase_rad": point.phase_rad,
            }
            for point in points
        ]
    )


@app.post("/api/v1/filters/zero-poles", response_model=ZeroPoleResponse)
async def evaluate_zero_poles(request: CoefficientsRequest) -> ZeroPoleResponse:
    result = zero_pole_analysis(request.b, request.a)
    return ZeroPoleResponse(**result)
