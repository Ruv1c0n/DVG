r"""
Лабораторная работа №3. Решение линейных интегральных уравнений 2-го рода
(Глазков Д.В., «Пакеты прикладных математических программ»).

    phi(t) = lambda * \int_a^b K(t,s) phi(s) ds + f(t)      (Фредгольм, варианты 1-9)
    phi(t) = lambda * \int_a^t K(t,s) phi(s) ds + f(t)      (Вольтерра, варианты 10-18)

Реализованы два метода:
  1) метод последовательных приближений  phi_{n+1} = A phi_n,
     останов по условию  max|phi_{n+1} - phi_n| < eps;
  2) сведение к СЛАУ  (I - Q) phi = f  (квадратурная формула трапеций).

Запуск:
    python lab1.py 5            # вариант 5
    python lab1.py 12 --n 200   # вариант 12, 200 разбиений
    python lab1.py 5 --save     # сохранить график в png
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
# совместимость NumPy 1.x/2.x
trapezoid = getattr(np, 'trapezoid', None) or np.trapz
LAMBDA = 1.0


# ----------------------------------------------------------------------------
# Построение сеточного оператора Q (формула трапеций)
# ----------------------------------------------------------------------------
def build_Q(kind, a, b, K, n, lam=LAMBDA):
    """Матрица Q размера (n+1)x(n+1): (Q phi)_j ~ lam * \\int K(t_j, s) phi(s) ds."""
    t = np.linspace(a, b, n + 1)
    h = (b - a) / n
    Kmat = K(t[:, None], t[None, :])            # K[j,k] = K(t_j, s_k)

    # веса трапеций для каждой строки j
    W = np.zeros((n + 1, n + 1))
    if kind == "F":                              # интеграл по [a, b]
        w = np.full(n + 1, h)
        w[0] = w[-1] = h / 2
        W[:] = w
    # Вольтерра: интеграл по [a, t_j]
    else:
        for j in range(1, n + 1):
            W[j, :j + 1] = h
            W[j, 0] = W[j, j] = h / 2
        # строка j=0: интеграл по пустому отрезку = 0
    return t, lam * Kmat * W


# ----------------------------------------------------------------------------
# Проверка условия сжимаемости оператора A (при lambda = 1)
# ----------------------------------------------------------------------------
def check_contraction(kind, a, b, K, lam=LAMBDA, m=400):
    t = np.linspace(a, b, m + 1)
    Kmat = K(t[:, None], t[None, :])
    if kind == "V":
        print(
            "Вольтерра: оператор A (или его степень) всегда сжимающий -> итерации сходятся.")
        return True
    M = np.max(np.abs(Kmat))
    B = np.sqrt(trapezoid(trapezoid(Kmat**2, t, axis=1), t))
    c_C = abs(lam) * M * (b - a)
    c_L2 = abs(lam) * B
    print(f"Проверка сжимаемости при lambda={lam}:")
    print(f"  C[a,b]:  M = {M:.5f},  |lam|*M*(b-a) = {c_C:.5f}  "
          f"({'<1, сжимающий' if c_C < 1 else '>=1, условие НЕ выполнено'})")
    print(f"  L2[a,b]: B = {B:.5f},  |lam|*B       = {c_L2:.5f}  "
          f"({'<1, сжимающий' if c_L2 < 1 else '>=1, условие НЕ выполнено'})")
    return (c_C < 1) or (c_L2 < 1)


# ----------------------------------------------------------------------------
# Метод 1: последовательные приближения
# ----------------------------------------------------------------------------
def solve_iterations(Q, f, eps=1e-10, max_iter=10000):
    phi = f.copy()                               # phi_0 = f (можно взять любое)
    for it in range(1, max_iter + 1):
        phi_new = Q @ phi + f                    # phi_{n+1} = A phi_n
        # rho(phi_{n+1}, phi_n) в метрике C
        diff = np.max(np.abs(phi_new - phi))
        phi = phi_new
        if diff < eps:
            return phi, it, True
        if not np.isfinite(diff) or diff > 1e100:
            return phi, it, False                # расходится
    return phi, max_iter, False


# ----------------------------------------------------------------------------
# Метод 2: сведение к СЛАУ  (I - Q) phi = f
# ----------------------------------------------------------------------------
def solve_slae(Q, f):
    I = np.eye(Q.shape[0])
    return np.linalg.solve(I - Q, f)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", type=int, nargs="?",
                    default=1, help="номер варианта 1..18")
    ap.add_argument("--n", type=int, default=100,
                    help="число разбиений отрезка")
    ap.add_argument("--eps", type=float, default=1e-10,
                    help="точность итераций")
    ap.add_argument("--save", action="store_true",
                    help="сохранить график в png")
    args = ap.parse_args()

    kind, a, b, K, f_fun = ("F", 0, pi, lambda t, s: np.cos(
        3 * t + 3 * s) / pi, lambda t: 2 * np.cos(3 * t))

    check_contraction(kind, a, b, K)

    t, Q = build_Q(kind, a, b, K, args.n)
    f = f_fun(t) * np.ones_like(t)

    if kind == "F":
        rho = max(abs(np.linalg.eigvals(Q)))
        print(f"Спектральный радиус сеточного оператора Q: {rho:.5f} "
              f"({'итерации должны сходиться' if rho < 1 else 'итерации расходятся'})")

    # --- метод 1 ---
    phi_it, n_it, ok = solve_iterations(Q, f, eps=args.eps)
    if ok:
        print(
            f"Метод последовательных приближений: сошёлся за {n_it} итераций.")
    else:
        print(
            f"Метод последовательных приближений: НЕ сошёлся ({n_it} итераций).")

    # --- метод 2 ---
    phi_sl = solve_slae(Q, f)
    print("Метод сведения к СЛАУ: решено.")

    if ok:
        print(
            f"max |phi_iter - phi_slae| = {np.max(np.abs(phi_it - phi_sl)):.3e}")

    # --- графики ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    if ok:
        ax[0].plot(t, phi_it, "b")
    else:
        ax[0].text(0.5, 0.5, "итерации расходятся",
                   ha="center", transform=ax[0].transAxes)
    ax[0].set_title("Последовательные приближения")
    ax[1].plot(t, phi_sl, "r")
    ax[1].set_title("Сведение к СЛАУ")
    if ok:
        ax[2].plot(t, phi_it, "b", lw=3, alpha=0.5,
                   label="послед. приближения")
    ax[2].plot(t, phi_sl, "r--", label="СЛАУ")
    ax[2].set_title("Сравнение")
    ax[2].legend()
    for x in ax:
        x.set_xlabel("t")
        x.set_ylabel(r"$\varphi(t)$")
        x.grid(True)
    fig.suptitle(f"Вариант {args.variant}, n = {args.n}")
    fig.tight_layout()
    if args.save:
        fn = f"lab3_variant{args.variant}.png"
        fig.savefig(fn, dpi=130)
        print("График сохранён:", fn)
    else:
        plt.show()


if __name__ == "__main__":
    main()
