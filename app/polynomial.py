"""多项式工具（核心逻辑手写）：升幂系数相乘、求值、求根。

系数一律按 z^-1 的非负幂次升序排列：
    [c0, c1, c2, ...] 表示 c0 + c1 z^-1 + c2 z^-2 + ...
求根时在 z 平面计算（先反转换到降幂 z 多项式，再用 Durand-Kerner 法）。
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence

from .errors import FilterError


def poly_multiply(p: Sequence[float], q: Sequence[float]) -> list[float]:
    """两个升幂实系数多项式卷积相乘。"""
    if not p or not q:
        raise FilterError("多项式不能为空")
    result = [0.0] * (len(p) + len(q) - 1)
    for i, ci in enumerate(p):
        for j, cj in enumerate(q):
            result[i + j] += ci * cj
    return result


def poly_from_roots(roots: Sequence[complex]) -> list[complex]:
    """由根构造升幂首一多项式：prod (1 - r z^-1)。"""
    poly: list[complex] = [1.0 + 0j]
    for root in roots:
        poly = poly_multiply(poly, [1.0, -root])
    return poly


def poly_eval_descending(coeffs: Sequence[complex], z: complex) -> complex:
    """Horner 法求值，coeffs 按 z 的降幂排列 [a0, a1, ..., an]。"""
    value = 0j
    for coef in coeffs:
        value = value * z + coef
    return value


def poly_derivative_descending(coeffs: Sequence[complex]) -> list[complex]:
    """降幂多项式求导。"""
    degree = len(coeffs) - 1
    if degree <= 0:
        return [0j]
    return [coeffs[k] * (degree - k) for k in range(degree)]


def _strip_trailing_zeros(coeffs: Sequence[float], tol: float = 1e-15) -> list[float]:
    """去掉 z^-1 升幂多项式末尾近零的高阶系数（如汉宁窗首尾严格为 0）。"""
    top = max(abs(c) for c in coeffs) if coeffs else 0.0
    if top == 0.0:
        raise FilterError("零多项式无法求根")
    threshold = tol * top
    end = len(coeffs)
    while end > 1 and abs(coeffs[end - 1]) < threshold:
        end -= 1
    return list(coeffs[:end])


def _aberth_iteration(
    coeffs: Sequence[complex],
    deriv: Sequence[complex],
    roots: list[complex],
) -> float:
    """Aberth-Ehrlich 单次同时迭代（Jacobi 式更新），返回本步最大相对修正量。

    z_i_new = z_i - 1 / ( f'(z_i)/f(z_i) - sum_{j!=i} 1/(z_i - z_j) )
    根间排斥项让互异近根各自收敛，也避免两个根抢占同一个位置。
    """
    degree = len(coeffs) - 1
    updates = list(roots)
    max_correction = 0.0
    for i in range(degree):
        root = roots[i]
        f_value = poly_eval_descending(coeffs, root)
        # 残差达到机器精度的根保持不动。阈值必须收紧到 1e-16 量级：
        # 重根附近 f 本身很小，若按系数量级提前停步，根位置还会偏差很多。
        residual_scale = max(1.0, abs(root)) ** degree
        if abs(f_value) <= 1e-16 * residual_scale:
            continue
        log_derivative = poly_eval_descending(deriv, root) / f_value
        offset = 0j
        for j in range(degree):
            if i == j:
                continue
            difference = root - roots[j]
            if abs(difference) < 1e-15:
                difference = 1e-15 * cmath.exp(1j * (i + 1) * (j + 1))
            offset += 1.0 / difference
        denominator = log_derivative - offset
        if abs(denominator) < 1e-300:
            continue
        correction = 1.0 / denominator
        # 限制步长，防止早期迭代飞出发散
        if abs(correction) > 0.5:
            correction *= 0.5 / abs(correction)
        updates[i] = root - correction
        relative = abs(correction) / max(1.0, abs(updates[i]))
        if relative > max_correction:
            max_correction = relative
    roots[:] = updates
    return max_correction


def _merge_genuine_multiple_roots(
    coeffs: Sequence[complex], deriv: Sequence[complex], roots: list[complex]
) -> None:
    """合并 Aberth 在线性收敛极限附近留下的真重根。

    互异近根与真重根的判别：m 重根 r0 同时让 f(r0)、f'(r0) 趋零，
    且簇心是 f^(m-1) 的单根。只有合并候选同时通过这两个残差检查才接受。
    """
    degree = len(coeffs) - 1
    parent = list(range(degree))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    scale = max(1.0, max(abs(r) for r in roots))
    # 本服务滤波器的互异零/极点间距都在 0.1 量级以上，而病态多项式
    # （如 (1+z^-1)^N）真重根的数值解间距在 1e-3 量级，阈值取中间
    threshold = 5e-3 * scale
    for i in range(degree):
        for j in range(i + 1, degree):
            if abs(roots[i] - roots[j]) < threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(degree):
        clusters.setdefault(find(i), []).append(i)

    coefficient_scale = max(abs(c) for c in coeffs)
    for members in clusters.values():
        multiplicity = len(members)
        if multiplicity == 1:
            continue
        center = sum(roots[i] for i in members) / multiplicity

        # 对 f^(m-1) 做 Newton，把簇心精修到真重根位置
        reduced = list(coeffs)
        for _ in range(multiplicity - 1):
            reduced = poly_derivative_descending(reduced)
        reduced_deriv = poly_derivative_descending(reduced)
        z = center
        for _ in range(80):
            denominator = poly_eval_descending(reduced_deriv, z)
            if abs(denominator) < 1e-300:
                break
            step = poly_eval_descending(reduced, z) / denominator
            z -= step
            if abs(step) <= 1e-15 * max(1.0, abs(z)):
                break

        residual_scale = coefficient_scale * max(1.0, abs(z)) ** degree
        f_residual = abs(poly_eval_descending(coeffs, z))
        fp_residual = abs(poly_eval_descending(deriv, z))
        # 真重根：f 和 f' 都为零；互异近根在簇心处至少有一个明显非零。
        # 展开系数的数值误差让 f 残差有 ~1e-14 的底噪，阈值相应放宽。
        if (
            f_residual <= 1e-10 * residual_scale
            and fp_residual <= 1e-8 * residual_scale
        ):
            for i in members:
                roots[i] = z


def roots_polynomial(coeffs_ascending: Sequence[float]) -> list[complex]:
    """对升幂 z^-1 排列的实系数多项式 c0 + c1 z^-1 + ... + cN z^-N 求根。

    乘 z^N 后为 z 平面降幂多项式 c0 z^N + c1 z^(N-1) + ... + cN，
    系数顺序与输入一致（无需反转），直接在此多项式上求根。

    采用手写的 Aberth-Ehrlich 同时迭代（Durand-Kerner 提供初值），
    不调用 numpy/scipy 的求根黑盒。返回按 (实部, 虚部) 排序的根。
    """
    if not coeffs_ascending:
        raise FilterError("系数为空，无法求根")
    if not all(isinstance(c, (int, float, complex)) for c in coeffs_ascending):
        raise FilterError("系数必须全部为数值")
    coeffs = [complex(c) for c in coeffs_ascending]
    # 剥掉 z^-1 升幂多项式中高阶的近零系数（如汉宁窗首尾严格为 0）
    descending = _strip_trailing_zeros(coeffs)
    degree = len(descending) - 1
    if degree == 0:
        return []

    # 初值：半径略小于 1 的圆上按黄金角分布，带正虚部偏移，
    # 避免初值落在实轴等对称位置上
    golden_angle = 2.399963229728653
    roots = [
        0.4 * cmath.exp(1j * golden_angle * k) + 0.01j * (k % 2)
        for k in range(degree)
    ]
    deriv = poly_derivative_descending(descending)

    # Durand-Kerner 粗迭代，把根送进各自的收敛域
    for iteration in range(300):
        max_correction = 0.0
        for i in range(degree):
            denominator = 1.0 + 0j
            for j in range(degree):
                if i != j:
                    factor = roots[i] - roots[j]
                    if abs(factor) < 1e-300:
                        # 根碰撞时加极小扰动，避免除零后 NaN 污染
                        factor = 1e-14 * cmath.exp(1j * (i + 1) * (j + 1))
                    denominator *= factor
            value = poly_eval_descending(descending, roots[i])
            correction = value / denominator
            # 限制单步修正量，避免初值不佳时直接飞出发散
            if abs(correction) > 1e6:
                correction *= 1e6 / abs(correction)
            roots[i] -= correction
            relative = abs(correction) / max(1.0, abs(roots[i]))
            if relative > max_correction:
                max_correction = relative
        # 个别根 NaN/Inf 时用独立几何序列重新播种
        for i, root in enumerate(roots):
            if not (math.isfinite(root.real) and math.isfinite(root.imag)):
                roots[i] = 0.6 * cmath.exp(1j * golden_angle * (i + iteration))
        if max_correction < 1e-6:
            break

    # Aberth 同时迭代：根间排斥项让互异近根各自分开、全部高精度收敛
    for _ in range(200):
        if _aberth_iteration(descending, deriv, roots) < 1e-14:
            break

    # 真重根在线性收敛极限附近仍有小间距，残差验证通过后才合并
    _merge_genuine_multiple_roots(descending, deriv, roots)

    roots.sort(key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    return roots
