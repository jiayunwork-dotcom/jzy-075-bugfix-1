"""音频低通滤波器设计与分析服务。

模块划分：
- windows:      窗函数生成（核心手写逻辑）
- fir:          FIR 窗函数法低通设计
- butterworth:  巴特沃斯模拟原型 + 预畸变 + 双线性变换（核心手写逻辑）
- polynomial:   多项式乘法 / 求值 / 求根（核心手写逻辑）
- phase:        相位展开（核心手写逻辑）
- analysis:     频响求值与零极点分析
- validation:   输入校验
- schemas:      请求 / 响应模型
- main:         FastAPI 路由
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
