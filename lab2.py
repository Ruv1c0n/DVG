"""
Лабораторная работа №6. Численные методы решения краевых задач для ОДУ.
Вариант 24:  y'' + 3y' + 2y = -2,   y(0) = 0,  y'(1) = 1,   n = 20.

Задача:      y'' + p*y' + q*y = f(x),  a <= x <= b
             alpha1*y(a) + beta1*y'(a) = gamma1
             alpha2*y(b) + beta2*y'(b) = gamma2

Разностная схема (6.3)-(6.4) из методички (односторонние разности для y'):
    (y[i-1] - 2y[i] + y[i+1])/h^2 + p*(y[i+1] - y[i])/h + q*y[i] = f[i]
    alpha1*y0 + beta1*(y1 - y0)/h = gamma1
    alpha2*yn + beta2*(yn - y[n-1])/h = gamma2

Методы решения сеточной задачи: правая прогонка, левая прогонка,
встречная прогонка, метод стрельбы.
"""
import math

# ---------------------------------------------------------------- данные варианта
p, q = 3.0, 2.0


def f(x):
    return -2.0


a, b, n = 0.0, 1.0, 20
alpha1, beta1, gamma1 = 1.0, 0.0, 0.0      # y(0) = 0
alpha2, beta2, gamma2 = 0.0, 1.0, 1.0      # y'(1) = 1

# Точное решение: Y = C1*y1 + C2*y2 + y0.
# Для другого варианта поменяйте ТОЛЬКО этот блок (y1, y2, y0 и их производные).
# Корни характеристического уравнения r^2 + 3r + 2 = 0:  r = -1, r = -2.


def y1(x): return math.exp(-x)
def dy1(x): return -math.exp(-x)
def y2(x): return math.exp(-2 * x)
def dy2(x): return -2 * math.exp(-2 * x)
def y0(x): return -1.0            # частное решение: 2*y0 = -2
def dy0(x): return 0.0


def exact_solution(xs):
    """Находит C1, C2 из граничных условий (определитель 2x2 по Крамеру)."""
    A11 = alpha1 * y1(a) + beta1 * dy1(a)
    A12 = alpha1 * y2(a) + beta1 * dy2(a)
    A21 = alpha2 * y1(b) + beta2 * dy1(b)
    A22 = alpha2 * y2(b) + beta2 * dy2(b)
    r1 = gamma1 - (alpha1 * y0(a) + beta1 * dy0(a))
    r2 = gamma2 - (alpha2 * y0(b) + beta2 * dy0(b))
    det = A11 * A22 - A12 * A21
    if abs(det) < 1e-14:
        raise ValueError(
            "Определитель равен нулю: краевая задача не имеет единственного решения")
    C1 = (r1 * A22 - A12 * r2) / det
    C2 = (A11 * r2 - r1 * A21) / det
    return [C1 * y1(x) + C2 * y2(x) + y0(x) for x in xs], C1, C2


# ---------------------------------------------------------------- сеточная задача
h = (b - a) / n
xs = [a + i * h for i in range(n + 1)]

# (6.5): y[i-1] + d[i]*y[i] + e[i]*y[i+1] = g[i],  i = 1..n-1
d = [0.0] * (n + 1)
e = [0.0] * (n + 1)
g = [0.0] * (n + 1)
for i in range(1, n):
    d[i] = -2.0 - p * h + q * h * h
    e[i] = 1.0 + p * h
    g[i] = h * h * f(xs[i])

# (6.6): y0 = chi1*y1 + mu1,  yn = chi2*y[n-1] + mu2
chi1 = beta1 / (beta1 - alpha1 * h) + 0.0
mu1 = -gamma1 * h / (beta1 - alpha1 * h)
chi2 = beta2 / (alpha2 * h + beta2)
mu2 = gamma2 * h / (alpha2 * h + beta2)


# ---------------------------------------------------------------- прямая прогонка
def forward_coeffs(upto):
    """y[i] = q[i]*y[i+1] + phi[i], i = 0..upto.  (6.7)-(6.10)"""
    Q = [0.0] * (n + 1)
    P = [0.0] * (n + 1)
    Q[0], P[0] = chi1, mu1
    for i in range(1, upto + 1):
        den = Q[i - 1] + d[i]
        Q[i] = -e[i] / den
        P[i] = (g[i] - P[i - 1]) / den
    return Q, P


def right_sweep():
    Q, P = forward_coeffs(n - 1)
    y = [0.0] * (n + 1)
    y[n] = (mu2 + chi2 * P[n - 1]) / (1.0 - chi2 * Q[n - 1])
    for i in range(n - 1, -1, -1):
        y[i] = Q[i] * y[i + 1] + P[i]
    return y


# ---------------------------------------------------------------- обратная прогонка
def backward_coeffs(downto):
    """y[i+1] = xi[i+1]*y[i] + eta[i+1], i = n-1..downto.  (6.11)-(6.13)"""
    X = [0.0] * (n + 1)
    H = [0.0] * (n + 1)
    X[n], H[n] = chi2, mu2
    for i in range(n - 1, downto - 1, -1):
        den = d[i] + e[i] * X[i + 1]
        X[i] = -1.0 / den
        H[i] = (g[i] - e[i] * H[i + 1]) / den
    return X, H


def left_sweep():
    X, H = backward_coeffs(1)
    y = [0.0] * (n + 1)
    y[0] = (mu1 + chi1 * H[1]) / (1.0 - chi1 * X[1])
    for i in range(0, n):
        y[i + 1] = X[i + 1] * y[i] + H[i + 1]
    return y


# ---------------------------------------------------------------- встречная прогонка
def counter_sweep(k=None):
    if k is None:
        k = n // 2
    Q, P = forward_coeffs(k)           # y[k]   = Q[k]*y[k+1]   + P[k]
    X, H = backward_coeffs(k + 1)      # y[k+1] = X[k+1]*y[k]   + H[k+1]
    y = [0.0] * (n + 1)
    y[k] = (Q[k] * H[k + 1] + P[k]) / (1.0 - Q[k] * X[k + 1])
    y[k + 1] = X[k + 1] * y[k] + H[k + 1]
    for i in range(k - 1, -1, -1):     # обратный ход правой прогонки
        y[i] = Q[i] * y[i + 1] + P[i]
    for i in range(k + 1, n):          # прямой ход левой прогонки
        y[i + 1] = X[i + 1] * y[i] + H[i + 1]
    return y


# ---------------------------------------------------------------- метод стрельбы
def shooting():
    U = [0.0] * (n + 1)
    V = [0.0] * (n + 1)
    W = [0.0] * (n + 1)
    U[0], U[1] = 0.0, 1.0
    V[0], V[1] = 1.0, 0.0
    W[0], W[1] = 0.0, 0.0
    for i in range(1, n):
        U[i + 1] = -(U[i - 1] + d[i] * U[i]) / e[i]
        V[i + 1] = -(V[i - 1] + d[i] * V[i]) / e[i]
        W[i + 1] = (g[i] - W[i - 1] - d[i] * W[i]) / e[i]
    # C2 = chi1*C1 + mu1
    # C1*Un + C2*Vn + Wn = chi2*(C1*U[n-1] + C2*V[n-1] + W[n-1]) + mu2
    # подставляем C2 и решаем относительно C1:
    A = (U[n] + chi1 * V[n]) - chi2 * (U[n - 1] + chi1 * V[n - 1])
    B = chi2 * (mu1 * V[n - 1] + W[n - 1]) + mu2 - mu1 * V[n] - W[n]
    C1 = B / A
    C2 = chi1 * C1 + mu1
    return [C1 * U[i] + C2 * V[i] + W[i] for i in range(n + 1)]


# ---------------------------------------------------------------- вывод
METHODS = {          # номера как в списке методов методички (стр. 61)
    "1": ("Метод прямой (правой) прогонки", right_sweep),
    "2": ("Метод обратной (левой) прогонки", left_sweep),
    "4": ("Метод встречной прогонки", counter_sweep),
    "5": ("Метод стрельбы", shooting),
}


def choose_methods():
    import sys
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    if choice is None:
        print("Выберите метод решения сеточной задачи:")
        for k, (name, _) in METHODS.items():
            print(f"  {k} - {name}")
        print("  0 - все методы сразу (сравнение)")
        choice = input("Ваш выбор: ").strip()
    if choice == "0":
        return list(METHODS)
    if choice not in METHODS:
        raise SystemExit(f"Нет метода с номером {choice!r}")
    return [choice]


def main():
    keys = choose_methods()
    ye, C1, C2 = exact_solution(xs)

    print(f"\nВариант 24: y'' + 3y' + 2y = -2, y(0)=0, y'(1)=1, n={n}, h={h}")
    print(f"Точное решение: y = {C1:.6f}*exp(-x) + {C2:.6f}*exp(-2x) - 1")
    print(f"d = {d[1]:.6f}, e = {e[1]:.6f}, g = {g[1]:.6f}")
    print(
        f"chi1 = {chi1:.6f}, mu1 = {mu1:.6f}, chi2 = {chi2:.6f}, mu2 = {mu2:.6f}\n")

    res = {k: METHODS[k][1]() for k in keys}

    head_row = f"{'i':>3} {'x_i':>6} {'точное':>12} " + \
        " ".join(f"{'метод '+k:>12}" for k in keys)
    print(head_row)
    print("-" * len(head_row))
    for i in range(n + 1):
        row = f"{i:>3} {xs[i]:>6.2f} {ye[i]:>12.6f} "
        row += " ".join(f"{res[k][i]:>12.6f}" for k in keys)
        print(row)

    print("\nМаксимальная погрешность max|y_i - y(x_i)|:")
    for k in keys:
        err = max(abs(res[k][i] - ye[i]) for i in range(n + 1))
        print(f"  {METHODS[k][0]:<34} {err:.3e}")

    plot(res, ye)


def plot(res, ye):
    import matplotlib.pyplot as plt

    marks = {"1": "o", "2": "s", "4": "^", "5": "x"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    xf = [a + (b - a) * j / 400 for j in range(401)]
    ax1.plot(xf, exact_solution(xf)[0], "k-", lw=2, label="точное решение")
    for k, y in res.items():
        ax1.plot(xs, y, marks[k], ms=7, mfc="none", label=METHODS[k][0])
    ax1.set_title("Точное и приближённое решения")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    ax1.legend()

    for k, y in res.items():
        ax2.plot(xs, [abs(y[i] - ye[i]) for i in range(n + 1)], marks[k] + "-",
                 ms=5, mfc="none", label=METHODS[k][0])
    ax2.set_title("Погрешность |y_i - y(x_i)|")
    ax2.set_xlabel("x")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    plt.show()          # окно с графиками открывает сама программа


if __name__ == "__main__":
    main()
