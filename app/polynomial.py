"""多项式工具（核心逻辑手写）：升幂系数相乘、求值、求根。

系数一律按 z^-1 的非负幂次升序排列：
    [c0, c1, c2, ...] 表示 c0 + c1 z^-1 + c2 z^-2 + ...
乘 z^N 后系数顺序天然成为 z 平面降幂多项式（无需反转），
求根直接在该多项式上用 Aberth-Ehrlich 法完成。
"""

from __future__ import annotations

import cmath
import itertools
import math
from collections.abc import Sequence

from .errors import FilterError

# 黄金角：初值在圆周上铺开时相邻根的角间距，避开实轴等对称位置。
_GOLDEN_ANGLE = 2.399963229728653


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


def _backward_error(coeffs: Sequence[complex], root: complex) -> float:
    """根的向后误差（归一化）：|f(z)| / sum |a_k| |z|^(n-k)。

    这是数值求根的标准后验指标：它有多大，等于把系数扰动多大幅度
    能让给定的 z 成为精确根。它对远离原点的大根同样有意义（不像
    |f(z)| 那样随 |z|^n 暴涨），因此用它而不是残差绝对值判断收敛。
    """
    degree = len(coeffs) - 1
    scale = 0.0
    power = 1.0
    for k in range(degree + 1):
        scale += abs(coeffs[k]) * power
        power *= abs(root)
    if scale == 0.0:
        return math.inf
    return abs(poly_eval_descending(coeffs, root)) / scale


def _derivative_n(coeffs: Sequence[complex], order: int) -> list[complex]:
    """降幂多项式求 order 阶导数。"""
    result = list(coeffs)
    for _ in range(order):
        result = poly_derivative_descending(result)
    return result


def _aberth_iterate(
    coeffs: Sequence[complex],
    deriv: Sequence[complex],
    roots: list[complex],
    step_cap: float,
    freeze_tol: float,
) -> int:
    """Aberth-Ehrlich 单次同时迭代（Jacobi 式更新），返回本步仍在移动的根数。

    z_i_new = z_i - 1 / ( f'(z_i)/f(z_i) - sum_{j!=i} 1/(z_i - z_j) )
    根间排斥项让互异近根各自收敛，也避免两个根抢占同一个位置。

    停止判据用逐根的向后误差（见 _backward_error），而不是统一残差：
    重根附近 f 本身趋零，统一残差会让根在没到位时就被冻结。
    步长上限取 max(1, |z|) 的固定比例，对大根是相对量、对近原点根
    是绝对量，避免旧实现里固定 0.5 把离群大根永久卡在远处。
    """
    degree = len(coeffs) - 1
    updates = list(roots)
    active = 0
    for i in range(degree):
        root = roots[i]
        f_value = poly_eval_descending(coeffs, root)
        if _backward_error(coeffs, root) <= freeze_tol:
            continue
        active += 1
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
        # 相对/绝对自适应限幅：早期防止飞出发散，后期不拖累大根回归
        limit = step_cap * max(1.0, abs(root))
        if abs(correction) > limit:
            correction *= limit / abs(correction)
        updates[i] = root - correction
    roots[:] = updates
    return active


def _refine_cluster_center(
    coeffs: Sequence[complex], members: Sequence[complex], multiplicity: int
) -> complex:
    """对 f^(m-1) 做 Newton，把候选 m 重根簇心精修到真重根位置。"""
    reduced = _derivative_n(coeffs, multiplicity - 1)
    reduced_deriv = poly_derivative_descending(reduced)
    z = sum(members) / multiplicity
    for _ in range(100):
        denominator = poly_eval_descending(reduced_deriv, z)
        if abs(denominator) < 1e-300:
            break
        step = poly_eval_descending(reduced, z) / denominator
        z -= step
        if abs(step) <= 1e-15 * max(1.0, abs(z)):
            break
    return z


def _taylor_coefficients(
    coeffs: Sequence[complex], center: complex, order: int
) -> list[complex]:
    """f 在 center 处的 Taylor 系数 c_k = f^(k)(center) / k!，k = 0..order。

    用对 (z - center) 的逐次综合除法（synthetic division）求得：
    每次除完的余式即当前阶系数，商多项式再除一次得到下一阶，
    全程只有乘加，不引入任何外部数值库。
    """
    remaining = list(coeffs)
    coefficients: list[complex] = []
    for _ in range(order + 1):
        if not remaining:
            coefficients.append(0j)
            continue
        quotient: list[complex] = []
        carry = 0j
        for value in remaining:
            carry = value + center * carry
            quotient.append(carry)
        coefficients.append(quotient[-1])
        remaining = quotient[:-1]
    return coefficients


def _multiplicity_test(
    coeffs: Sequence[complex], center: complex, multiplicity: int, tol: float
) -> tuple[bool, float, float]:
    """检验 center 是否为 multiplicity 重根（局部 Taylor 系数判据）。

    在 center 处展开 f(center+h) = sum c_k h^k。若 center 真的是 m 重根，
    c_0..c_{m-1} 都应只有机器精度量级，而 c_m 是 Taylor 展式的首个主项、
    明显非零；若参与合并的其实是 m 个互异近根，则某个低阶系数
    c_k（k<m）是主项量级，c_m 反而被 (间距)^(m-k) 压小。

    判据为相对比值 max_{k<m}|c_k| / |c_m| <= tol。这是一个纯粹的局部
    解析量，不依赖簇外根的距离，因此不会把"离高阶重根不远的互异根簇"
    （如散开在 -1 周围、实际各自独立的 IIR 极点）错判成重根。实测
    真重根该比值最大约 5e-10，最近的互异极点对也在 1e-3 以上。
"""
    taylor = _taylor_coefficients(coeffs, center, multiplicity)
    low_order = max(abs(taylor[k]) for k in range(multiplicity))
    leading = abs(taylor[multiplicity])
    accepted = low_order <= tol * max(leading, 1e-300)
    return accepted, low_order, leading


def _merge_multiple_roots(coeffs: Sequence[complex], roots: list[complex]) -> None:
    """识别并合并真重根（如巴特沃斯分子 (1+z^-1)^N 的 N 重根 -1）。

    Aberth 对真重根只是线性收敛，N 越大停得越早、散开范围越大，
    不能用固定距离阈值判定（互异近根会被误并，并簇的残差又会把
    离群根拽到错误位置）。

    两步走：
    1. 全部根同根：对 f^(N-1) 精修簇心，局部 Taylor 判据通过则整簇
       一次合并（巴特沃斯分子走这条，与阶数无关，N=16 也能合到精确 -1）；
    2. 否则按根间距由近到远贪心并簇，每次合并都通过两条检验：
       - 成员近邻：簇内每个根到最近同伴的距离都在簇间距阈值内，
         排除彼此相隔 O(1) 的假簇（它们的 f^(m-1) Newton 只会
         收敛到多项式里别处的真重根）；
       - "重数恰好为 m"的局部 Taylor 判据（见 _multiplicity_test）。
    """
    degree = len(coeffs) - 1
    if degree < 2:
        return

    # Taylor 比值阈值取 1e-7：实测真重根该比值最大约 5e-10（高阶、
    # 小首项系数时综合除法噪声上浮），互异极点对即便相距最近也在 1e-3 以上。
    tolerance = 1e-7
    # 成员近邻阈值：一个 m 重根簇里，每个成员到簇内最近同伴的距离
    # 就是簇的自然间距。对 f^(m-1) 做 Newton 时初始点取成员均值，
    # 收敛域只覆盖该间距量级；成员彼此相隔 O(1) 的"簇"精修后会跑到
    # 多项式里别处的真重根上（例如互异共轭对被吸到 -1），必须拒绝。
    proximity = 0.3

    def is_local_cluster(member_roots: Sequence[complex]) -> bool:
        if len(member_roots) < 2:
            return True
        for root in member_roots:
            nearest = min(
                abs(root - other)
                for other in member_roots
                if other is not root
            )
            if nearest > proximity:
                return False
        return True

    # 1) 全部根同根的快路径（N 重根的常见情形）
    full_center = _refine_cluster_center(coeffs, roots, degree)
    accepted, _low, _lead = _multiplicity_test(
        coeffs, full_center, degree, tolerance
    )
    if accepted:
        roots[:] = [full_center] * degree
        return

    # 2) 局部贪心合并
    parent = list(range(degree))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs = sorted(
        (abs(roots[i] - roots[j]), i, j)
        for i, j in itertools.combinations(range(degree), 2)
    )
    for _distance, i, j in pairs:
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            continue
        member_roots = [roots[x] for x in range(degree) if find(x) in (root_i, root_j)]
        multiplicity = len(member_roots)
        # 成员先得真的凑成一簇（彼此近邻），否则 f^(m-1) 的 Newton
        # 精修只会跑到多项式里别处的真重根上去。
        if not is_local_cluster(member_roots):
            continue
        center = _refine_cluster_center(coeffs, member_roots, multiplicity)
        accepted, _low, _lead = _multiplicity_test(
            coeffs, center, multiplicity, tolerance
        )
        if accepted:
            parent[root_i] = root_j

    clusters: dict[int, list[int]] = {}
    for i in range(degree):
        clusters.setdefault(find(i), []).append(i)
    for members in clusters.values():
        if len(members) > 1:
            member_roots = [roots[i] for i in members]
            center = _refine_cluster_center(
                coeffs, member_roots, len(members)
            )
            for i in members:
                roots[i] = center


def _newton_polish_with_error(
    coeffs: Sequence[complex], deriv: Sequence[complex], root: complex
) -> tuple[complex, float]:
    """对单根做 Newton 精修，返回 (最好的根, 对应向后误差)。

    只在向后误差确实下降时接受步长。真重根处 Newton 发散，遇到步长
    让误差变大就停在原根，不破坏 _merge_multiple_roots 已确认的重根
    结构。
    """
    best = root
    best_error = _backward_error(coeffs, root)
    z = root
    for _ in range(30):
        f_value = poly_eval_descending(coeffs, z)
        f_deriv = poly_eval_descending(deriv, z)
        if abs(f_deriv) < 1e-300:
            break
        step = f_value / f_deriv
        candidate = z - step
        candidate_error = _backward_error(coeffs, candidate)
        if candidate_error < best_error:
            best, best_error = candidate, candidate_error
        if abs(step) <= 1e-15 * max(1.0, abs(candidate)):
            break
        # 误差开始反弹（撞上重根或进入坏收敛域），保留目前最好的位置
        if candidate_error > 10.0 * _backward_error(coeffs, z):
            break
        z = candidate
    return best, best_error


def _enforce_real_coefficient_symmetry(
    coeffs: Sequence[complex], roots: list[complex]
) -> list[complex]:
    """把实系数多项式的根投影到严格的实根 / 共轭对结构。

    Aberth 在浮点数下算出的共轭对会有 1e-16..1e-8 的不对称，
    重根合并后虚部也可能残留符号噪声。处理顺序：

    1. 已经被 _merge_multiple_roots 精确合并的重根（如巴特沃斯
       的 N 重根 -1）按槽位原样保留：它们的位置由 f^(m-1) 的根
       精修确定，再做单根 Newton 反而会沿重根根域发散、把整簇
       根带偏。实重根顺手清掉符号级虚部噪声。
    2. 互异根先分实根、再配共轭对。不能只按 Aberth 给出的虚部大小
    分：奇数阶 IIR 的实极点可能停在虚部 1e-7，而贴轴的共轭极点虚部
    可以小到 1e-3。做法是先对"虚部不大"（<=1e-3 相对）的候选逐个在
    实轴上 Newton 精修，首步相对修正量 <=2e-3 的才认定实根摘出。
    实测真单实根（含两个相距很近的互异实根）首步不超过 ~1e-11，
    贴轴共轭对最小也有 ~5e-3。
    3. 剩下的根在上下半平面之间做总共轭失配最小的一一匹配
    （手写匈牙利法），对上半平面成员作 Newton 精修再镜像，不顺序
    贪心配对，避免相邻共轭对交叉配错。
    """
    degree = len(coeffs) - 1
    deriv = poly_derivative_descending(coeffs)
    result: list[complex] = []
    remaining = set(range(degree))

    def real_root_confirmed(root: complex) -> tuple[complex, float, bool]:
        """在实轴上精修候选根，返回 (根, 向后误差, 是否确为实根)。"""
        candidate = complex(root.real, 0.0)
        polished, error = _newton_polish_with_error(coeffs, deriv, candidate)
        f_value = poly_eval_descending(coeffs, polished)
        f_deriv = poly_eval_descending(deriv, polished)
        if abs(f_deriv) > 1e-300:
            first_step = abs(f_value / f_deriv) / max(1.0, abs(polished))
        else:
            first_step = math.inf
        is_real = error <= 1e-11 and first_step <= 2e-3
        return complex(polished.real, 0.0), error, is_real

    def close(a: complex, b: complex) -> bool:
        # 合并后的重根槽位位置由同一个簇心赋值，距离恒为 0；实测互异
        # 近根（如汉宁窗 FIR 互反根对）最近也有约 5e-13 的相对间距，
        # 阈值取 1e-13 只认合并产物、不碰真正的互异近根。
        return abs(a - b) <= 1e-13 * max(1.0, abs(a), abs(b))

    # 1) 已确认的重根：把指向同一位置的槽位整组摘出，全部保留。
    #    这一步必须先于配对，否则一组 m 重根会被当成 m 个互异根
    #    两两配对、镜像后只剩 2 个位置。
    for i in list(remaining):
        group = [j for j in remaining if close(roots[i], roots[j])]
        if len(group) < 2:
            continue
        for j in group:
            root = roots[j]
            if abs(root.imag) <= 1e-12:
                result.append(complex(root.real, 0.0))
            else:
                result.append(root)
            remaining.discard(j)

    # 2) 实单根：对虚部 <=1e-3（相对）的候选在实轴上精修确认。
    for i in list(remaining):
        root = roots[i]
        if abs(root.imag) > 1e-3 * max(1.0, abs(root)):
            continue
        polished, _error, is_real = real_root_confirmed(root)
        if is_real:
            result.append(polished)
            remaining.remove(i)

    # 3) 其余互异根在上下半平面之间做总共轭失配最小的一一匹配
    #    （手写匈牙利法），对上半平面成员作 Newton 精修，再镜像出
    #    严格共轭的另一半。不能顺序贪心配对：相邻共轭对在实轴方向
    #    挨得很近时会交叉配错。
    def is_lower_half(index: int) -> bool:
        return roots[index].imag < 0.0

    lower = sorted(
        (i for i in remaining if is_lower_half(i)),
        key=lambda x: roots[x].real,
    )
    upper = sorted(
        (i for i in remaining if not is_lower_half(i)),
        key=lambda x: roots[x].real,
    )

    unmatched: list[int] = []
    pairs_idx: list[tuple[int, int]] = []
    if len(lower) == len(upper):
        pairs_idx = _minimum_conjugate_matching(roots, lower, upper)
        matched: set[int] = set()
        for i, j in pairs_idx:
            matched.add(i)
            matched.add(j)
        unmatched = [i for i in remaining if i not in matched]
    else:
        # 上下半平面根数不等只可能来自数值噪声或求根仍有偏差，
        # 对没配上的根逐个再做一次实轴确认，仍不成立的保持原位置，
        # 由后面的共轭对称投影兜底。
        unmatched = sorted(remaining)

    for i, j in pairs_idx:
        start = roots[j]  # 上半平面成员
        polished, _error = _newton_polish_with_error(coeffs, deriv, start)
        upper_root = complex(polished.real, abs(polished.imag))
        result.extend((upper_root, upper_root.conjugate()))

    # 兜底（实系数多项式理论上不会走到）：逐个再做实轴确认；
    # 确认为实根的压到实轴，其余的原样保留（配对已保证主体共轭）。
    for i in unmatched:
        polished, _error, is_real = real_root_confirmed(roots[i])
        if is_real:
            result.append(polished)
        else:
            result.append(roots[i])

    return result


def _minimum_conjugate_matching(
    roots: Sequence[complex], lower: list[int], upper: list[int]
) -> list[tuple[int, int]]:
    """下半平面根与上半平面根的总共轭失配最小一一匹配。

    代价矩阵 cost[a][b] = |z_lower,a - conj(z_upper,b)|，用手写的
    匈牙利法（Kuhn-Munkres）求最小权完美匹配，不引入外部数值库。
    两侧根数相等（k <= 16），规模小，直接 O(k^3)。
    """
    size = len(lower)
    if size == 0:
        return []

    cost = [
        [abs(roots[a] - roots[b].conjugate()) for b in upper]
        for a in lower
    ]
    # 匈牙利法（行约简 + 增广路），标准一维标号实现。
    # match_column[列] = 匹配的行（1-based），第 0 列是增广路的虚拟起点，
    # 因此数组长度需要 size + 1 个可匹配列再加一个虚拟列。
    inf = math.inf
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    way = [0] * (size + 1)
    match_column = [0] * (size + 1)
    for row in range(1, size + 1):
        match_column[0] = row
        column = 0
        minv = [inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            matched_row = match_column[column]
            delta = inf
            next_column = 0
            for candidate in range(1, size + 1):
                if not used[candidate]:
                    current = (
                        cost[matched_row - 1][candidate - 1]
                        - u[matched_row]
                        - v[candidate]
                    )
                    if current < minv[candidate]:
                        minv[candidate] = current
                        way[candidate] = column
                    if minv[candidate] < delta:
                        delta = minv[candidate]
                        next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[match_column[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minv[candidate] -= delta
            column = next_column
            if match_column[column] == 0:
                break
        # 沿增广路翻转匹配
        while column != 0:
            previous = way[column]
            match_column[column] = match_column[previous]
            column = previous

    result: list[tuple[int, int]] = []
    for column in range(1, size + 1):
        row = match_column[column]
        result.append((lower[row - 1], upper[column - 1]))
    return result


def roots_polynomial(coeffs_ascending: Sequence[float]) -> list[complex]:
    """对升幂 z^-1 排列的实系数多项式 c0 + c1 z^-1 + ... + cN z^-N 求根。

    乘 z^N 后为 z 平面降幂多项式 c0 z^N + c1 z^(N-1) + ... + cN，
    系数顺序与输入一致（无需反转），直接在此多项式上求根。

    采用手写的 Aberth-Ehrlich 同时迭代，不调用 numpy/scipy 的求根黑盒。
    流程：系数归一化 -> 单位圆黄金角初值 -> Aberth 迭代（向后误差
    逐根冻结、相对自适应步长）-> 真重根验证合并 -> 实系数共轭对称
    约束投影与单根 Newton 精修。返回按 (实部, 虚部) 排序的根。
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

    # 归一化到首项尺度，统一容差口径，也避免系数过小放大舍入误差
    coefficient_scale = max(abs(c) for c in descending)
    descending = [c / coefficient_scale for c in descending]
    deriv = poly_derivative_descending(descending)

    # 初值直接铺在单位圆上：本服务的滤波器零、极点都在单位圆量级
    # （FIR 互为倒数的根对也以 |z|=1 为中心），比固定半径 0.4 更贴近
    # 全部根的天然位置，Aberth 从这里出发十几个迭代即可全部到位。
    roots = [
        cmath.exp(1j * (_GOLDEN_ANGLE * k + 0.1)) + 0.01j * (k % 2)
        for k in range(degree)
    ]

    # Aberth 同时迭代：根间排斥项让互异近根各自分开、全部高精度收敛
    for _ in range(500):
        active = _aberth_iterate(
            descending, deriv, roots, step_cap=0.5, freeze_tol=1e-15
        )
        if active == 0:
            break

    # 真重根在线性收敛极限附近散开，残差验证通过后才合并
    _merge_multiple_roots(descending, roots)

    # 实系数 -> 根严格实/共轭，并对单根做最后一轮 Newton 精修
    roots = _enforce_real_coefficient_symmetry(descending, roots)

    roots.sort(key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    return roots
