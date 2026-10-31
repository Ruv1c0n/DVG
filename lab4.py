"""
Лабораторная работа №8. Решение уравнения Лапласа методом сеток.

Вариант 16.   u_xx + u_yy = 0  в области D,
    граница Г:  |y| = 9 - x,  -3 <= x <= 3   (+ вертикальные отрезки x = -3 и x = 3,
                                              замыкающие область: трапеция с вершинами
                                              (-3, +-12) и (3, +-6))
    u|Г = |x| + 0.5*|y|

Сетка:  x_m = -3 + m*h,  y_n = -12 + n*h,   h1 = h2 = h = 1/K.
Так как границы |y| = 9 - x имеют наклон +-1 и h1 = h2, они проходят ТОЧНО
через узлы сетки (при h, делящем 24), поэтому значения u в граничных узлах
берутся из условия без погрешности переноса.

Разностное уравнение (8.3) при h1 = h2:
    u[m,n] = (u[m-1,n] + u[m+1,n] + u[m,n-1] + u[m,n+1]) / 4
Система решается итерационно:
    1 - метод Зейделя,  2 - метод наискорейшего спуска.

Запуск:
    python lab4.py                 - параметры спросит программа
    python lab4.py <метод> <K>     - например:  python lab4.py 1 10
        K = 1/h:  K=10 -> h=0.1 (как в методичке),  K=5 -> h=0.2 (быстрее)
"""
import sys
import time
import numpy as np

# ------------------------------------------------------------------ область
XMIN, XMAX = -3, 3          # границы по x
def YMAX_AT_X(X): return 9 - X  # |y| = 9 - x


def boundary_value(x, y):
    return abs(x) + 0.5 * abs(y)


def build_grid(K):
    """Целочисленные координаты (в единицах h) исключают ошибки округления."""
    nx, ny = 6 * K + 1, 24 * K + 1
    xs = np.array([-3.0 + i / K for i in range(nx)])
    ys = np.array([-12.0 + j / K for j in range(ny)])
    inside = np.zeros((nx, ny), dtype=bool)      # внутренние узлы
    bound = np.zeros((nx, ny), dtype=bool)       # граничные узлы
    for i in range(nx):
        X = i - 3 * K
        for j in range(ny):
            Y = j - 12 * K
            lim = 9 * K - X
            if abs(X) == 3 * K and abs(Y) <= lim:
                bound[i, j] = True
            elif abs(X) < 3 * K and abs(Y) == lim:
                bound[i, j] = True
            elif abs(X) < 3 * K and abs(Y) < lim:
                inside[i, j] = True
    u = np.zeros((nx, ny))
    for i, j in zip(*np.nonzero(bound)):
        u[i, j] = boundary_value(xs[i], ys[j])
    u[inside] = u[bound].mean()                  # начальное приближение
    return xs, ys, inside, bound, u


# ------------------------------------------------------------------ методы
def seidel(u0, inside, K, eps, max_iter):
    """Метод Зейделя: новые значения используются сразу."""
    u = u0.tolist()
    nx = len(u)
    # для каждого столбца i - диапазон внутренних j
    ranges = {}
    for i in range(1, nx - 1):
        js = np.nonzero(inside[i])[0]
        if len(js):
            ranges[i] = (int(js[0]), int(js[-1]))
    hist = []
    for it in range(1, max_iter + 1):
        dmax = 0.0
        for i, (lo, hi) in ranges.items():
            col, um, up = u[i], u[i - 1], u[i + 1]
            for j in range(lo, hi + 1):
                new = 0.25 * (um[j] + up[j] + col[j - 1] + col[j + 1])
                d = abs(new - col[j])
                if d > dmax:
                    dmax = d
                col[j] = new
        hist.append(dmax)
        if dmax < eps:
            break
    return np.array(u), it, hist


def steepest_descent(u0, inside, eps, max_iter):
    """Наискорейший спуск для A u = b, A = -Лаплас (симметрична, положительно определена)."""
    u = u0.copy()
    mask = inside.astype(float)

    def neighbors(a):
        s = np.zeros_like(a)
        s[1:-1, :] += a[:-2, :] + a[2:, :]
        s[:, 1:-1] += a[:, :-2] + a[:, 2:]
        return s

    hist = []
    for it in range(1, max_iter + 1):
        r = (neighbors(u) - 4.0 * u) * mask          # r = b - A u
        Ar = (4.0 * r - neighbors(r)) * mask
        rr = np.sum(r * r)
        if rr == 0.0:
            break
        alpha = rr / np.sum(r * Ar)
        u += alpha * r
        rmax = np.max(np.abs(alpha * r))
        hist.append(rmax)
        if rmax < eps:
            break
    return u, it, hist


def direct_reference(u, inside):
    """Прямое решение той же разностной системы (для проверки итераций)."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
    except ImportError:
        return None
    idx = -np.ones(inside.shape, dtype=int)
    pts = list(zip(*np.nonzero(inside)))
    for k, (i, j) in enumerate(pts):
        idx[i, j] = k
    A = sp.lil_matrix((len(pts), len(pts)))
    b = np.zeros(len(pts))
    for k, (i, j) in enumerate(pts):
        A[k, k] = 4.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if inside[ii, jj]:
                A[k, idx[ii, jj]] = -1.0
            else:
                b[k] += u[ii, jj]
    sol = spl.spsolve(A.tocsr(), b)
    ref = u.copy()
    for k, (i, j) in enumerate(pts):
        ref[i, j] = sol[k]
    return ref


# ------------------------------------------------------------------ вывод
def print_table(xs, ys, u, inside, bound, K):
    """Таблица значений в узлах с шагом 1 по x и 2 по y (внутри области)."""
    cols = list(range(0, len(xs), K))
    print("\nПриближённое решение u(x, y) (шаг таблицы: по x - 1, по y - 2;  '.' - вне области)")
    print(f"{'y \\ x':>7}" + "".join(f"{xs[i]:>8.1f}" for i in cols))
    for j in range(len(ys) - 1, -1, -2 * K):
        row = f"{ys[j]:>7.1f}"
        for i in cols:
            row += f"{u[i, j]:>8.3f}" if (inside[i, j]
                                          or bound[i, j]) else f"{'.':>8}"
        print(row)


def plot(xs, ys, u, inside, bound, hist, name, K):
    import matplotlib.pyplot as plt

    region = inside | bound
    Z = np.ma.masked_where(~region, u).T            # строки - y, столбцы - x
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        13, 8), gridspec_kw={"width_ratios": [1, 2]})

    cf = ax1.contourf(xs, ys, Z, levels=20, cmap="viridis")
    ax1.contour(xs, ys, Z, levels=20, colors="k", linewidths=0.3)
    xb = [-3, 3, 3, -3, -3]
    yb = [-12, -6, 6, 12, -12]
    ax1.plot(xb, yb, "r-", lw=1.5, label="граница Г")
    fig.colorbar(cf, ax=ax1, shrink=0.8, label="u")
    ax1.set_aspect("equal")
    ax1.set_title(f"Решение u(x, y), h = {1.0 / K:g}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.semilogy(range(1, len(hist) + 1), hist, "b-")
    ax2.set_title(f"Сходимость: {name}")
    ax2.set_xlabel("номер итерации")
    ax2.set_ylabel("max |u_new - u_old|")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    plt.show()


def ask_params():
    if len(sys.argv) >= 3:
        return sys.argv[1], int(sys.argv[2])
    print("Итерационный метод:\n  1 - метод Зейделя\n  2 - метод наискорейшего спуска")
    m = ""
    while m not in ("1", "2"):
        m = input("Выберите метод: ").strip()
    k = input("K = 1/h (10 -> h=0.1, 5 -> h=0.2) [10]: ").strip()
    return m, int(k) if k else 10


def main():
    method, K = ask_params()
    h = 1.0 / K
    xs, ys, inside, bound, u0 = build_grid(K)
    n_in, n_b = int(inside.sum()), int(bound.sum())

    print(f"\nВариант 16: u_xx + u_yy = 0, область |y| <= 9 - x, -3 <= x <= 3, u|Г = |x| + 0.5|y|")
    print(f"Шаги h1 = h2 = {h:g};  узлов: внутренних {n_in}, граничных {n_b}")

    eps = 1e-8
    t = time.time()
    if method == "1":
        name = "метод Зейделя"
        u, it, hist = seidel(u0, inside, K, eps, 200000)
    else:
        name = "метод наискорейшего спуска"
        u, it, hist = steepest_descent(u0, inside, eps, 200000)
    print(f"{name}: {it} итераций, {time.time() - t:.1f} с, критерий max|u_new - u_old| < {eps:g}")

    print_table(xs, ys, u, inside, bound, K)

    # проверки
    umin, umax = u[inside].min(), u[inside].max()
    bmin, bmax = u[bound].min(), u[bound].max()
    print(f"\nПринцип максимума: u внутри [{umin:.4f}, {umax:.4f}] в пределах граничных [{bmin:.4f}, {bmax:.4f}]"
          f" -> {'выполнен' if bmin - 1e-9 <= umin and umax <= bmax + 1e-9 else 'НАРУШЕН'}")
    res = max(abs(u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j])
              for i, j in zip(*np.nonzero(inside)))
    print(f"Невязка разностных уравнений: max|сумма соседей - 4u| = {res:.2e}")
    ref = direct_reference(u0, inside)
    if ref is not None:
        print(
            f"Отличие от прямого решения той же системы: {np.max(np.abs(u - ref)):.2e}")

    plot(xs, ys, u, inside, bound, hist, name, K)


if __name__ == "__main__":
    main()
