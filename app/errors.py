"""统一的业务异常。

所有参数校验、设计与求值环节的可预期错误都抛 ``FilterError``，
由 FastAPI 的异常处理器转成 HTTP 400（JSON）。
"""


class FilterError(ValueError):
    """滤波器设计 / 分析环节的输入或计算错误。"""
