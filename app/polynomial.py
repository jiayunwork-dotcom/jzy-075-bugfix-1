"""多项式工具（核心逻辑手写）：升幂系数相乘、求值、求根。

系数一律按 z^-1 的非负幂次升序排列：
    [c0, c1, c2, ...] 表示 c0 + c1 z^-1 + c2 z^-2 + ...
求根时在 z 平面计算：[c0..cN] 直接看作降幂 z 多项式 c0 z^N + ... + cN
（系数顺序与输入一致，无需反转），用 Aberth-Ehrlich 同时迭代求出全部根。

保证输出可信的结构设计：
- 实系数多项式从共轭对称的初值出发，迭代中逐轮强制共轭，输出的
  复根天然共轭成对；多组确定性初值配合"整组根乘回系数"的后向
  误差检验，避免伪平衡与重根邻域的假收敛；
- 数值上的重根（如巴特沃斯分子 (1+z^-1)^N，双精度下根在 -1 周围
  散开 ~eps^(1/N)，单根位置本不可分辨）做"吸附回代验证"的合并：
  只有 (z - c)^m 除尽多项式到机器精度时才把簇内根吸附到中心 c，
  保证合并结果仍是原系数（浮点误差内）的真根，零极点重组频响不受影响。
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence

from .errors import FilterError

_MACHINE_EPS = 2.220446049250313e-16


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


def _strip_leading_zeros(coeffs: Sequence[complex], tol: float = 1e-15) -> list[complex]:
    """去掉降幂 z 多项式首项近零的系数（实际次数比系数个数暗示的低）。"""
    top = max(abs(c) for c in coeffs)
    threshold = tol * top
    start = 0
    while start < len(coeffs) - 1 and abs(coeffs[start]) < threshold:
        start += 1
    return list(coeffs[start:])


def _initial_guesses(coeffs: Sequence[complex], degree: int) -> list[complex]:
    """共轭对称的初值：z^degree = -const/lead 的根（模长为根模几何平均）。

    角度取 (2k+1)pi/degree，点集关于实轴对称；实系数多项式从对称初值
    出发，Aberth 迭代保持共轭对称。叠加一个同样关于实轴对称的半径
    抖动，避免完全对称的布局让迭代停滞。
    """
    radius = _geometric_mean_radius(coeffs, degree)
    guesses = []
    for k in range(degree):
        angle = math.pi * (2 * k + 1) / degree
        jitter = 1.0 + 0.02 * math.cos(3.0 * angle) + 0.01 * math.cos(5.0 * angle)
        guesses.append(radius * jitter * cmath.exp(1j * angle))
    return guesses


def _geometric_mean_radius(coeffs: Sequence[complex], degree: int) -> float:
    """根模的几何平均 |const/lead|^(1/degree)，作为初值圆的基准半径。"""
    ratio = abs(coeffs[-1] / coeffs[0])
    radius = ratio ** (1.0 / degree) if ratio > 0.0 else 1.0
    return min(max(radius, 1e-150), 1e150)


def _spiral_guesses(radius: float, degree: int) -> list[complex]:
    """非对称螺旋初值（黄金角 + 确定性半径抖动）。

    全实根多项式配共轭对称初值时，一对共轭迭代点可能卡在非根的
    伪平衡点上；非对称布局打破该平衡，作为残差检验后的重试布局。
    """
    golden_angle = 2.399963229728653
    return [
        radius
        * (1.0 + 0.13 * math.sin(1.7 * k + 0.3))
        * cmath.exp(1j * golden_angle * k)
        for k in range(degree)
    ]


def _aberth_iteration(
    coeffs: Sequence[complex],
    deriv: Sequence[complex],
    roots: list[complex],
    converged: list[bool],
) -> float:
    """Aberth-Ehrlich 单次同时迭代（Jacobi 式更新），返回本步最大相对修正量。

    z_i_new = z_i - 1 / ( f'(z_i)/f(z_i) - sum_{j!=i} 1/(z_i - z_j) )
    根间排斥项让互异近根各自收敛，也避免两个根抢占同一个位置。
    Jacobi 式（同一快照统一更新）保证实系数时共轭对称不被破坏。
    """
    degree = len(roots)
    updates = list(roots)
    max_correction = 0.0
    for i in range(degree):
        if converged[i]:
            continue
        root = roots[i]
        f_value = poly_eval_descending(coeffs, root)
        if f_value == 0:
            converged[i] = True
            continue
        log_derivative = poly_eval_descending(deriv, root) / f_value
        offset = 0j
        for j in range(degree):
            if i == j:
                continue
            difference = root - roots[j]
            if difference == 0:
                # 精确相撞只可能来自病态输入，给一个人为微扰继续迭代
                difference = _MACHINE_EPS * (1 + 1j)
            offset += 1.0 / difference
        denominator = log_derivative - offset
        if denominator == 0:
            updates[i] = root + _MACHINE_EPS * max(1.0, abs(root)) * (1 + 1j)
            continue
        correction = 1.0 / denominator
        # 步长限制在根自身尺度的数倍内：允许大步追赶远处的根，
        # 但杜绝根被一步甩到 1e8 量级那种发散
        limit = 4.0 * max(1.0, abs(root))
        if abs(correction) > limit:
            correction *= limit / abs(correction)
        updates[i] = root - correction
        relative = abs(correction) / max(1.0, abs(updates[i]))
        if relative <= 8.0 * _MACHINE_EPS:
            converged[i] = True
        if relative > max_correction:
            max_correction = relative
    roots[:] = updates
    return max_correction


def _aberth_roots(
    coeffs: Sequence[complex], roots: list[complex], paired: bool
) -> list[complex]:
    """Aberth-Ehrlich 同时迭代的主循环（就地更新并返回根列表）。

    单根以修正量 <= 8 eps（相对）判收敛并冻结；重根簇内修正量降到
    噪声底后会随机徘徊、不再趋势性下降，连续多轮无显著改善即停，
    簇的精确位置由后续的验证合并统一处理。

    paired=True（实系数 + 共轭对称初值）时，每轮迭代后把共轭对
    强制取为精确的共轭——对称性不被浮点噪声侵蚀，输出的复零点
    共轭配对到机器精度。
    """
    degree = len(coeffs) - 1
    deriv = poly_derivative_descending(coeffs)
    converged = [False] * degree
    # 对称初值的共轭对下标：(k, degree-1-k)；奇数次时中间项初值为实
    partners = [(k, degree - 1 - k) for k in range(degree // 2)] if paired else []
    reference = float("inf")
    stale_rounds = 0
    for _ in range(1000):
        max_correction = _aberth_iteration(coeffs, deriv, roots, converged)
        for rep, partner in partners:
            roots[partner] = roots[rep].conjugate()
            converged[partner] = converged[rep]
        if paired and degree % 2 == 1:
            middle = (degree - 1) // 2
            roots[middle] = roots[middle].real + 0j
        if all(converged):
            break
        if max_correction < 0.5 * reference:
            reference = max_correction
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= 100:
                break
    return roots


def _set_backward_error(coeffs: Sequence[complex], roots: Sequence[complex]) -> float:
    """根集的系数级后向误差：c0 * prod(z - z_i) 展开后与存储系数比较。

    这是"这组根是否真是该系数多项式的根"的判据：重根附近逐根残差
    不可靠（重根邻域内处处残差小，根数都可能对错），但整组根乘回去
    的系数必须吻合，否则零极点重组频响对不上。
    """
    product = [coeffs[0]]
    for root in roots:
        product = _convolve_complex(product, [1.0 + 0j, -root])
    top = max(abs(c) for c in coeffs)
    return max(abs(c - p) for c, p in zip(coeffs, product, strict=True)) / top


def _converged_roots(coeffs: Sequence[complex]) -> list[complex]:
    """多组确定性初值依次尝试，返回系数级后向误差达机器精度量级的一组根。

    每组初值迭代收敛后先做重根验证合并，再整组乘回系数检验
    （重根邻域内逐根残差会假性合格，必须整组对账）。
    第一组初值共轭对称（实系数时逐轮强制共轭）：成功时输出天然
    共轭成对。若检验未过（典型情形：全实根时共轭初值对卡在伪平衡
    点上），换非对称螺旋初值重试；此时共轭配对由收敛精度保证。
    多组均未达标时返回后向误差最小的一组（病态输入下的尽力结果）。
    """
    degree = len(coeffs) - 1
    real_coeffs = all(c.imag == 0 for c in coeffs)
    base_radius = _geometric_mean_radius(coeffs, degree)
    attempts = [
        (_initial_guesses(coeffs, degree), real_coeffs),
        (_spiral_guesses(base_radius, degree), False),
        (_spiral_guesses(0.31 * base_radius, degree), False),
        (_spiral_guesses(3.17 * base_radius, degree), False),
    ]
    best_roots: list[complex] | None = None
    best_error = float("inf")
    for guesses, paired in attempts:
        roots = _aberth_roots(coeffs, guesses, paired)
        _merge_multiple_roots(coeffs, roots)
        error = _set_backward_error(coeffs, roots)
        if error <= 1e-9:
            return roots
        if error < best_error:
            best_error = error
            best_roots = roots
    assert best_roots is not None  # attempts 非空，必然有候选
    return best_roots


def _quadratic_roots(coeffs: Sequence[complex]) -> list[complex]:
    """实系数二次多项式直接求解（判别式为负时给精确的共轭对）。"""
    c0, c1, c2 = (coeffs[0].real, coeffs[1].real, coeffs[2].real)
    discriminant = c1 * c1 - 4.0 * c0 * c2
    if discriminant >= 0.0:
        sqrt_d = math.sqrt(discriminant)
        # 数值稳定形式：先算远离抵消的那个根，再用韦达定理算另一个
        q = -(c1 + math.copysign(sqrt_d, c1)) / 2.0
        if q == 0.0:
            return [0j, 0j]
        return [complex(q / c0, 0.0), complex(c2 / q, 0.0)]
    real = -c1 / (2.0 * c0)
    imag = math.sqrt(-discriminant) / (2.0 * c0)
    return [complex(real, imag), complex(real, -imag)]


def _refine_cluster_center(
    coeffs: Sequence[complex], multiplicity: int, start: complex
) -> complex:
    """把簇心精修到 m 重根位置：m 重根是 f^(m-1) 的单根，对其做 Newton。"""
    reduced = list(coeffs)
    for _ in range(multiplicity - 1):
        reduced = poly_derivative_descending(reduced)
    reduced_deriv = poly_derivative_descending(reduced)
    z = start
    for _ in range(100):
        denominator = poly_eval_descending(reduced_deriv, z)
        if denominator == 0:
            break
        step = poly_eval_descending(reduced, z) / denominator
        z -= step
        if abs(step) <= 4.0 * _MACHINE_EPS * max(1.0, abs(z)):
            break
    return z


def _convolve_complex(
    p: Sequence[complex], q: Sequence[complex]
) -> list[complex]:
    """两个复系数多项式卷积（降幂/升幂解释无关）。"""
    result = [0j] * (len(p) + len(q) - 1)
    for i, ci in enumerate(p):
        for j, cj in enumerate(q):
            result[i + j] += ci * cj
    return result


def _snap_relative_error(
    coeffs: Sequence[complex], center: complex, multiplicity: int
) -> float:
    """把 multiplicity 重根吸附到 center 后，系数层面的相对扰动量。

    先做 m 次 (z - center) 综合除法得到商 Q，再重组
    P_snap = (z - center)^m * Q，返回 max|P - P_snap| / max|P|。
    扰动在机器精度量级，说明吸附等价于给系数加浮点噪声级的修改，
    即吸附后的根仍是原系数（浮点误差内）的真根——这正是
    "零极点重组频响与系数直接代入一致"所需要的后向稳定性。
    """
    quotient = list(coeffs)
    for _ in range(multiplicity):
        next_quotient = [quotient[0]]
        for k in range(1, len(quotient) - 1):
            next_quotient.append(quotient[k] + center * next_quotient[-1])
        quotient = next_quotient
    snapped = list(quotient)
    for _ in range(multiplicity):
        snapped = _convolve_complex(snapped, [1.0 + 0j, -center])
    top = max(abs(c) for c in coeffs)
    return max(abs(c - s) for c, s in zip(coeffs, snapped, strict=True)) / top


def _cluster_components(roots: Sequence[complex], threshold: float) -> list[list[int]]:
    """并查集：两两距离小于 threshold 的根连成同一簇，返回下标分组。"""
    degree = len(roots)
    parent = list(range(degree))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(degree):
        for j in range(i + 1, degree):
            if abs(roots[i] - roots[j]) < threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    clusters: dict[int, list[int]] = {}
    for i in range(degree):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _try_merge_cluster(
    coeffs: Sequence[complex], roots: list[complex], members: list[int]
) -> bool:
    """尝试把 members 指向的簇吸附到公共中心，成功返回 True。

    判别分三步，缺一不可：
    1. Newton 精修 f^(m-1) 的根作为候选中心；
    2. 中心必须仍在簇附近——Newton 可能跳到 f^(m-1) 的远处根，
       把本不属于该处的根错误地吸附过去；
    3. 吸附回代验证：(z - center)^m 必须除尽多项式到机器精度
       （系数相对扰动 < 1e-9）。互异近根通不过该检验，不会被误并；
       通过则说明吸附后向稳定，零极点重组频响不受影响。

    实系数多项式的真重根若在实轴上，候选中心先按实根验证，
    保证吸附结果严格为实、共轭配对不被破坏。
    """
    multiplicity = len(members)
    centroid = sum(roots[i] for i in members) / multiplicity
    spread = max(abs(roots[i] - centroid) for i in members)
    center = _refine_cluster_center(coeffs, multiplicity, centroid)
    if abs(center - centroid) > 2.0 * spread + 1e-6 * max(1.0, abs(centroid)):
        return False
    candidates = []
    if abs(center.imag) <= 1e-9 * max(1.0, abs(center)):
        candidates.append(complex(center.real, 0.0))
    candidates.append(center)
    for candidate in candidates:
        if _snap_relative_error(coeffs, candidate, multiplicity) <= 1e-9:
            for i in members:
                roots[i] = candidate
            return True
    return False


def _merge_multiple_roots(coeffs: Sequence[complex], roots: list[complex]) -> None:
    """把数值上的重根簇吸附到精修后的公共中心（见 _try_merge_cluster）。

    真重根的数值散开半径 ~eps^(1/m)*scale（m=16 时约 0.1），簇检测
    阈值取 0.5*scale 足以覆盖，误并一律由回代验证拦截。链式连接可能
    把真重根簇与邻近独立根混进同一组件：整体合并失败时细分再试。
    """
    degree = len(roots)
    if degree < 2:
        return
    scale = max(1.0, max(abs(r) for r in roots))
    for members in _cluster_components(roots, 0.5 * scale):
        if len(members) < 2:
            continue
        if _try_merge_cluster(coeffs, roots, members):
            continue
        sub_components = _cluster_components([roots[i] for i in members], 0.15 * scale)
        for sub in sub_components:
            if len(sub) >= 2:
                _try_merge_cluster(coeffs, roots, [members[p] for p in sub])


def _enforce_conjugate_symmetry(roots: list[complex]) -> None:
    """实系数多项式的根集做共轭对称投影（就地）。

    真根集天然共轭对称；非对称初值迭代（或簇内噪声徘徊）会让
    配对偏差停在病态量级（近重根时可达 1e-6 上下）。把每个正虚部根
    与最近的负虚部根配对并取平均，得到精确的共轭对；近实轴的根
    虚部直接清零。配对偏差本身很小（收敛根 ~1e-13），投影移动的
    距离对系数而言是机器精度量级的扰动，不影响零极点重组频响。
    """
    scale = max(1.0, max(abs(r) for r in roots))
    for i, z in enumerate(roots):
        if abs(z.imag) <= 1e-9 * max(1.0, abs(z)):
            roots[i] = complex(z.real, 0.0)
    used = [False] * len(roots)
    for i, z in enumerate(roots):
        if used[i] or z.imag == 0.0:
            continue
        best, best_gap = None, 1e-6 * scale
        for j, w in enumerate(roots):
            if j == i or used[j]:
                continue
            gap = abs(w - z.conjugate())
            if gap < best_gap:
                best, best_gap = j, gap
        if best is None:
            continue
        center = (z + roots[best].conjugate()) / 2.0
        roots[i] = center
        roots[best] = center.conjugate()
        used[i] = used[best] = True


def roots_polynomial(coeffs_ascending: Sequence[float]) -> list[complex]:
    """对升幂 z^-1 排列的实系数多项式 c0 + c1 z^-1 + ... + cN z^-N 求根。

    乘 z^N 后为 z 平面降幂多项式 c0 z^N + c1 z^(N-1) + ... + cN，
    系数顺序与输入一致（无需反转），直接在此多项式上求根。

    采用手写的 Aberth-Ehrlich 同时迭代（共轭对称初值 + 重根验证合并），
    不调用 numpy/scipy 的求根黑盒。返回按 (实部, 虚部) 排序的根。
    """
    if not coeffs_ascending:
        raise FilterError("系数为空，无法求根")
    if not all(isinstance(c, (int, float, complex)) for c in coeffs_ascending):
        raise FilterError("系数必须全部为数值")
    coeffs = [complex(c) for c in coeffs_ascending]
    # 剥掉 z^-1 升幂多项式中高阶的近零系数（如汉宁窗首尾严格为 0），
    # 以及 z 多项式首项的近零系数（实际次数更低）
    descending = _strip_leading_zeros(_strip_trailing_zeros(coeffs))
    degree = len(descending) - 1
    if degree == 0:
        return []

    # 归一化到最大系数模为 1，改善迭代的浮点表现（根不受整体缩放影响）
    top = max(abs(c) for c in descending)
    normalized = [c / top for c in descending]

    if degree == 1:
        roots = [-normalized[1] / normalized[0]]
    elif degree == 2 and all(c.imag == 0 for c in normalized):
        roots = _quadratic_roots(normalized)
    else:
        roots = _converged_roots(normalized)
        if all(c.imag == 0 for c in normalized):
            _enforce_conjugate_symmetry(roots)

    roots.sort(key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    return roots
