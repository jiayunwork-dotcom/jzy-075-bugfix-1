# 低通滤波器设计与分析服务

常驻后端服务：FIR（窗函数法）/ IIR（巴特沃斯 + 双线性变换）系数设计、
单位圆频响求值、零极点分析。三个 HTTP 接口，JSON 进 JSON 出。

Python 3.12 + FastAPI，科学计算只借标准库（`math` / `cmath`）完成核心算法；
窗函数生成、双线性预畸变、相位展开、多项式求根均为手写，可单独测试，
没有整段调用黑盒设计函数。

## 模块划分

| 文件 | 职责 |
| --- | --- |
| `app/windows.py` | 矩形 / 汉宁（hann、hanning 同义）/ 汉明窗，手写 |
| `app/fir.py` | FIR 窗函数法低通：理想冲激响应 × 窗 |
| `app/butterworth.py` | 巴特沃斯原型极点、频率预畸变、双线性变换，手写 |
| `app/polynomial.py` | 多项式乘法 / 求值，Aberth-Ehrlich 求根，手写 |
| `app/phase.py` | 主值相位与一维相位展开，手写 |
| `app/analysis.py` | 频响（dB + 展开相位）、零极点（到单位圆距离 + 稳定性） |
| `app/validation.py` | 阶数 / 截止 / 系数 / 频点统一校验 |
| `app/schemas.py` | Pydantic 请求响应模型 |
| `app/main.py` | FastAPI 路由与统一错误处理 |

## 约定

- **阶数**：正整数，1..16（超过 16 拒绝）；抽头数 = 阶数 + 1。
- **截止频率**：相对奈奎斯特的归一化值，严格开区间 `(0, 1)`；
  对应数字角频率 `w = cutoff * pi`（rad/采样）。0 和 1（奈奎斯特）都拒绝。
- **系数格式**：FIR / IIR 统一返回 `{"b": [...], "a": [...]}`，
  按 z⁻¹ 升幂排列；IIR 的 `a[0]` 归一化为 1，FIR 的 `a = [1.0]`。
- **窗类型**：`rectangular`（`rect`/`boxcar`）、`hann`（`hanning` 同义）、
  `hamming`，别名大小写与首尾空格不敏感；不认识的名字返回 400。

## 接口

### `POST /api/v1/filters/design`

```json
// FIR
{"type": "fir", "order": 12, "cutoff": 0.3, "window": "hamming"}
// IIR
{"type": "iir", "order": 4, "cutoff": 0.25}
```

### `POST /api/v1/filters/frequency`

频率点为数字角频率（rad/采样），范围 `[0, pi]`；
返回每点 `magnitude_db` 与展开后的 `phase_rad`。

```json
{"b": [1.0], "a": [1.0, -0.7], "frequencies": [0.0, 0.5, 1.0]}
```

### `POST /api/v1/filters/zero-poles`

```json
{"b": [1.0], "a": [1.0, -0.7]}
```

返回零极点列表（实部 / 虚部 / 模 / 到单位圆距离 / 是否在单位圆上），
系统总增益 `gain`，以及整体 `stable`（存在单位圆外极点即为 `false`）。

所有业务错误（非法阶数 / 截止、空频点数组、频点或系数里掺非数值等）
统一返回 HTTP 400，形如 `{"error": "中文原因"}`。

## 运行

```bash
docker build -t filter-service .
docker run --rm -p 8000:8000 filter-service
# 或本地
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

自动化测试盯住以下关系：

- FIR 系数首尾对称（浮点容差内），奇数 / 偶数阶切换时抽头数与对称中心跟着变；
- 预畸变与反预畸变互相反推；并构造一份"漏做预畸变"的对照实现，
  其真实 -3dB 交叉点明显偏离目标，防止用未预畸变的值糊弄；
- FIR 通带内展开相位斜率 = -order/2（群延时）；
- 同一组系数，频响直接代入与"零极点求根后重组"两条路径在截止附近
  幅度 dB 与衰减趋势一致；
- IIR 极点全部在单位圆内、`a[0]=1`、指定截止处约 -3.01 dB；
- 单位圆外极点明确标 `stable=false`，圆上极点单独标注；
- 阶数 0 / 负数 / >16、截止 0 / ≥1、未知窗名、空频点、非数值元素等一律拒绝。
