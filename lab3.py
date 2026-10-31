"""
Лабораторная работа №7. Численные методы решения краевых задач
для уравнения теплопроводности.

Вариант 17:   u_t = u_xx + t,   0 <= x <= 1
              u(x, 0) = x,      u(0, t) = 0,   u(1, t) = t

Требуется найти решение при t = t0 в точках x_i = i*h, i = 0..n:
  1) сеточное (разностная схема + метод прогонки),
  2) методом Фурье (ряд до третьего члена).

Запуск:
    python lab3.py                 - параметры спросит программа
    python lab3.py <схема> <метод> <набор n,m,t0>
        схема: 1 - неявная, 2 - Кранка-Николсона, 3 - повышенного порядка, 0 - все
        метод: 1 - прямая прогонка, 2 - обратная, 3 - встречная, 4 - стрельба
        набор: 1..6 (таблица n, m, t0 из методички)
"""
import math
import sys

# ------------------------------------------------------------ данные варианта


def f(x, t):                 # правая часть уравнения
    return t


def f_xx(x, t):              # вторая производная f по x (нужна для схемы 3)
    return 0.0


def phi(x):                  # начальное условие u(x, 0)
    return x


# граничные условия: alpha*u + beta*u_x = gamma(t)
alpha1, beta1 = 1.0, 0.0
alpha2, beta2 = 1.0, 0.0
def gamma1(t): return 0.0    # u(0, t) = 0
def gamma2(t): return t      # u(1, t) = t


PARAMS = {                   # набор: (n, m, t0)
    1: (20, 25, 1.0),
    2: (15, 25, 1.0),
    3: (15, 15, 0.5),
    4: (15, 25, 1.5),
    5: (25, 25, 1.5),
    6: (15, 25, 0.5),
}

# ------------------------------------------------------------ метод Фурье
# Замена u = w + v, w = x*t (w(0,t) = 0, w(1,t) = t).
# Для v:  v_t = v_xx + (t - x),  v(x,0) = x,  v(0,t) = v(1,t) = 0.
# Собственные функции e_k = sin(k*pi*x), lambda_k = -(k*pi)^2.
#   Phi_k = 2(-1)^(k+1)/(k pi)                     - разложение x
#   g_k(t) = A_k * t + B_k,  A_k = 2(1-(-1)^k)/(k pi),  B_k = 2(-1)^k/(k pi)
#   v_k' = lambda_k v_k + g_k,  v_k(0) = Phi_k


def fourier_coeff(k, t):
    lam = -(k * math.pi) ** 2
    s = (-1) ** k
    Phi = -2.0 * s / (k * math.pi)
    A = 2.0 * (1 - s) / (k * math.pi)
    B = 2.0 * s / (k * math.pi)
    E = math.exp(lam * t)
    return Phi * E + A * (E - 1 - lam * t) / lam ** 2 + B * (E - 1) / lam


def fourier(x, t, terms=3):
    return x * t + sum(fourier_coeff(k, t) * math.sin(k * math.pi * x)
                       for k in range(1, terms + 1))


# ------------------------------------------------------------ прогонки
# y[i-1] + d[i]*y[i] + e[i]*y[i+1] = g[i],  i = 1..n-1
# y[0] = chi1*y[1] + mu1,   y[n] = chi2*y[n-1] + mu2
def forward_coeffs(d, e, g, chi1, mu1, upto):
    Q = [0.0] * len(d)
    P = [0.0] * len(d)
    Q[0], P[0] = chi1, mu1
    for i in range(1, upto + 1):
        den = Q[i - 1] + d[i]
        Q[i] = -e[i] / den
        P[i] = (g[i] - P[i - 1]) / den
    return Q, P


def backward_coeffs(d, e, g, chi2, mu2, n, downto):
    X = [0.0] * len(d)
    H = [0.0] * len(d)
    X[n], H[n] = chi2, mu2
    for i in range(n - 1, downto - 1, -1):
        den = d[i] + e[i] * X[i + 1]
        X[i] = -1.0 / den
        H[i] = (g[i] - e[i] * H[i + 1]) / den
    return X, H


def right_sweep(d, e, g, chi1, mu1, chi2, mu2, n):
    Q, P = forward_coeffs(d, e, g, chi1, mu1, n - 1)
    y = [0.0] * (n + 1)
    y[n] = (mu2 + chi2 * P[n - 1]) / (1.0 - chi2 * Q[n - 1])
    for i in range(n - 1, -1, -1):
        y[i] = Q[i] * y[i + 1] + P[i]
    return y


def left_sweep(d, e, g, chi1, mu1, chi2, mu2, n):
    X, H = backward_coeffs(d, e, g, chi2, mu2, n, 1)
    y = [0.0] * (n + 1)
    y[0] = (mu1 + chi1 * H[1]) / (1.0 - chi1 * X[1])
    for i in range(n):
        y[i + 1] = X[i + 1] * y[i] + H[i + 1]
    return y


def counter_sweep(d, e, g, chi1, mu1, chi2, mu2, n):
    k = n // 2
    Q, P = forward_coeffs(d, e, g, chi1, mu1, k)
    X, H = backward_coeffs(d, e, g, chi2, mu2, n, k + 1)
    y = [0.0] * (n + 1)
    y[k] = (Q[k] * H[k + 1] + P[k]) / (1.0 - Q[k] * X[k + 1])
    y[k + 1] = X[k + 1] * y[k] + H[k + 1]
    for i in range(k - 1, -1, -1):
        y[i] = Q[i] * y[i + 1] + P[i]
    for i in range(k + 1, n):
        y[i + 1] = X[i + 1] * y[i] + H[i + 1]
    return y


def shooting(d, e, g, chi1, mu1, chi2, mu2, n):
    """Метод стрельбы: решение разностной задачи y = C1*U + C2*V + W,
    где U, V, W удовлетворяют разностному уравнению с начальными
    условиями U0=0,U1=1; V0=1,V1=0; W0=0,W1=0. Константы C1, C2
    находятся из обоих граничных условий."""
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
    # левое условие y0 = chi1*y1 + mu1  =>  C2 = chi1*C1 + mu1
    # правое условие yn = chi2*y[n-1] + mu2 - подставляем C2 и решаем по C1
    A = (U[n] + chi1 * V[n]) - chi2 * (U[n - 1] + chi1 * V[n - 1])
    B = chi2 * (mu1 * V[n - 1] + W[n - 1]) + mu2 - mu1 * V[n] - W[n]
    C1 = B / A
    C2 = chi1 * C1 + mu1
    return [C1 * U[i] + C2 * V[i] + W[i] for i in range(n + 1)]


SWEEPS = {
    "1": ("прямая прогонка", right_sweep),
    "2": ("обратная прогонка", left_sweep),
    "3": ("встречная прогонка", counter_sweep),
    "4": ("метод стрельбы", shooting),
}

SCHEMES = {
    "1": "неявная схема (sigma = 1)",
    "2": "схема Кранка - Никольсона (sigma = 1/2)",
    "3": "схема повышенного порядка аппроксимации",
}


# ------------------------------------------------------------ разностные схемы
def solve_grid(scheme, sweep, n, m, t0):
    h = 1.0 / n
    tau = t0 / m
    xs = [i * h for i in range(n + 1)]
    u = [phi(x) for x in xs]                       # слой j = 0

    # коэффициенты граничных условий (7.7)-(7.8), производная заменена на (u1-u0)/h
    chi1 = beta1 / (beta1 - alpha1 * h)
    chi2 = beta2 / (alpha2 * h + beta2)

    for j in range(m):
        t_old = j * tau
        t_new = (j + 1) * tau
        d = [0.0] * (n + 1)
        e = [1.0] * (n + 1)
        g = [0.0] * (n + 1)

        if scheme == "1":                          # (7.12)
            for i in range(1, n):
                d[i] = -(2.0 + h * h / tau)
                g[i] = -(h * h * f(xs[i], t_new) + h * h / tau * u[i])

        elif scheme == "2":                        # (7.13)
            for i in range(1, n):
                d[i] = -(2.0 + 2.0 * h * h / tau)
                g[i] = -(h * h * (f(xs[i], t_new) + f(xs[i], t_old))
                         + 2.0 * h * h / tau * u[i]
                         + u[i - 1] - 2.0 * u[i] + u[i + 1])

        else:                                      # схема повышенного порядка
            sigma = 0.5 - h * h / (12.0 * tau)
            tm = t_old + tau / 2.0
            for i in range(1, n):
                d[i] = -(2.0 + h * h / (tau * sigma))
                phi_i = f(xs[i], tm) + h * h / 12.0 * f_xx(xs[i], tm)
                g[i] = (-h * h / (tau * sigma) * u[i]
                        - (1.0 - sigma) / sigma *
                        (u[i - 1] - 2.0 * u[i] + u[i + 1])
                        - h * h / sigma * phi_i)

        mu1 = -gamma1(t_new) * h / (beta1 - alpha1 * h)
        mu2 = gamma2(t_new) * h / (alpha2 * h + beta2)
        u = sweep(d, e, g, chi1, mu1, chi2, mu2, n)

    return xs, u


# ------------------------------------------------------------ интерфейс
def ask(prompt, options):
    while True:
        s = input(prompt).strip()
        if s in options:
            return s
        print("  Нет такого варианта, повторите.")


def get_choices():
    a = sys.argv[1:]
    if len(a) >= 3:
        return a[0], a[1], int(a[2])
    print("Схема:")
    for k, v in SCHEMES.items():
        print(f"  {k} - {v}")
    print("  0 - все три схемы (сравнение)")
    scheme = ask("Выберите схему: ", ["0", "1", "2", "3"])
    print("Метод решения сеточной задачи:")
    for k, (name, _) in SWEEPS.items():
        print(f"  {k} - {name}")
    method = ask("Выберите метод: ", list(SWEEPS))
    print("Набор параметров (n, m, t0):")
    for k, (n_, m_, t_) in PARAMS.items():
        print(f"  {k} - n={n_}, m={m_}, t0={t_}")
    pset = int(ask("Выберите набор: ", [str(k) for k in PARAMS]))
    return scheme, method, pset


def main():
    scheme, method, pset = get_choices()
    n, m, t0 = PARAMS[pset]
    h, tau = 1.0 / n, t0 / m
    schemes = list(SCHEMES) if scheme == "0" else [scheme]
    sweep_name, sweep = SWEEPS[method]

    print(f"\nВариант 17: u_t = u_xx + t, u(x,0)=x, u(0,t)=0, u(1,t)=t")
    print(f"n={n}, m={m}, t0={t0}, h={h:.5f}, tau={tau:.5f}, метод: {sweep_name}\n")

    xs = [i * h for i in range(n + 1)]
    four = [fourier(x, t0, 3) for x in xs]          # ряд Фурье, 3 члена
    ref = [fourier(x, t0, 200) for x in xs]         # эталон для проверки

    res = {}
    for s in schemes:
        res[s] = solve_grid(s, sweep, n, m, t0)[1]

    head = f"{'i':>3} {'x_i':>7} {'Фурье(3)':>11} " + \
        " ".join(f"{'схема '+s:>11}" for s in schemes)
    print(head)
    print("-" * len(head))
    for i in range(n + 1):
        row = f"{i:>3} {xs[i]:>7.4f} {four[i]:>11.6f} "
        row += " ".join(f"{res[s][i]:>11.6f}" for s in schemes)
        print(row)

    print("\nМаксимальные отклонения:")
    print(
        f"  Фурье (3 члена) от эталона (200 членов): {max(abs(four[i]-ref[i]) for i in range(n+1)):.3e}")
    for s in schemes:
        d1 = max(abs(res[s][i] - four[i]) for i in range(n + 1))
        d2 = max(abs(res[s][i] - ref[i]) for i in range(n + 1))
        print(
            f"  схема {s}: от Фурье(3) {d1:.3e}, от эталона {d2:.3e}   [{SCHEMES[s]}]")

    plot(xs, four, ref, res, t0)


def plot(xs, four, ref, res, t0):
    import matplotlib.pyplot as plt

    marks = {"1": "o", "2": "s", "3": "^"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    xf = [j / 400 for j in range(401)]
    ax1.plot(xf, [fourier(x, t0, 200) for x in xf], "k-",
             lw=1.5, label="эталон (200 членов ряда)")
    ax1.plot(xs, four, "r--", lw=2, label="ряд Фурье (3 члена)")
    for s, y in res.items():
        ax1.plot(xs, y, marks[s], ms=7, mfc="none", label=SCHEMES[s])
    ax1.set_title(f"Решение при t = {t0}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.grid(alpha=0.3)
    ax1.legend()

    for s, y in res.items():
        ax2.plot(xs, [y[i] - four[i] for i in range(len(xs))], marks[s] + "-",
                 ms=5, mfc="none", label=f"схема {s} - Фурье(3)")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("Отклонение сеточного решения от ряда Фурье")
    ax2.set_xlabel("x")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
