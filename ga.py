"""
Вариант: N=27, Q=0, Y=26
  V = 1 + ((17*27 + 7*0 + 26) mod 20) = 6
  Функция Швефеля, d = 5, диапазон [-500, 500]
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------- ПАРАМЕТРЫ ----------------------
D = 5                  # размерность (из варианта)
LOW, HIGH = -500.0, 500.0
POP_SIZE = 60          # размер популяции
GENERATIONS = 300      # число поколений
ELITE = 2              # сколько лучших переходит без изменений
PC = 0.9               # вероятность кроссинговера
PM = 0.3             # вероятность мутации одного гена
SIGMA = 150.0         # СКО мутации (10% от диапазона 1000)
ALPHA = 0.5            # параметр BLX-альфа
TOURNAMENT = 3         # размер турнира при селекции

OPT_X = 420.96874636   # координата глобального минимума
OPT_F = 0.0


# ---------------------- ФУНКЦИЯ ШВЕФЕЛЯ ----------------------
def schwefel(x):
    """f(x) = 418.9829*d - sum(x_i * sin(sqrt(|x_i|))).
    Глобальный минимум f*=0 в точке x_i = 420.96874636."""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


# ---------------------- ОПЕРАТОРЫ ГА ----------------------
def init_population(pop_size, rng):
    """Случайная инициализация: 60 точек равномерно в [LOW, HIGH]^D."""
    return rng.uniform(LOW, HIGH, size=(pop_size, D))


def evaluate(pop):
    """Считаем фитнес для каждой особи. Чем меньше — тем лучше."""
    return np.array([schwefel(ind) for ind in pop])


def tournament(pop, fit, rng):
    """Турнирная селекция: берём TOURNAMENT случайных, возвращаем лучшего."""
    idx = rng.integers(0, len(pop), TOURNAMENT)
    return pop[idx[np.argmin(fit[idx])]].copy()


def crossover_blx(p1, p2, rng):
    """BLX-α: потомок в расширенном интервале [min-α*d, max+α*d]."""
    cmin = np.minimum(p1, p2)
    cmax = np.maximum(p1, p2)
    diff = cmax - cmin
    lo = cmin - ALPHA * diff
    hi = cmax + ALPHA * diff
    return rng.uniform(lo, hi)


def mutate(ind, rng, sigma):
    """Гауссовская мутация: каждый ген мутирует с вероятностью PM."""
    mask = rng.random(D) < PM
    return ind + mask * rng.normal(0.0, sigma, D)


def repair(ind):
    """Обработка границ: обрезаем координаты в [LOW, HIGH]."""
    return np.clip(ind, LOW, HIGH)


# ---------------------- ОСНОВНОЙ ЦИКЛ ГА ----------------------
def run_ga(seed, pop_size=POP_SIZE, generations=GENERATIONS, adaptive=False):
    """
    Один запуск ГА.
    Возвращает: лучшую точку, её фитнес, историю поколений, число вызовов f.

    adaptive=True включает правило 1/5 для масштаба мутации (доп. задание).
    """
    rng = np.random.default_rng(seed)
    sigma = SIGMA
    n_evals = 0

    # 1. Инициализация
    pop = init_population(pop_size, rng)
    fit = evaluate(pop)
    n_evals += pop_size

    best_x = pop[np.argmin(fit)].copy()
    best_f = float(fit.min())
    history = []

    # 2. Цикл по поколениям
    for gen in range(generations):
        history.append({
            "gen": gen,
            "best": float(fit.min()),
            "mean": float(fit.mean()),
            "std": float(fit.std()),
        })

        # Элитизм: копируем лучших без изменений
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:ELITE]]
        new_fit = [float(fit[i]) for i in order[:ELITE]]

        n_success = 0
        n_trials = 0
        prev_mean = float(fit.mean())

        # Генерируем потомков, пока не заполним популяцию
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fit, rng)
            p2 = tournament(pop, fit, rng)

            # Кроссинговер с вероятностью PC
            if rng.random() < PC:
                c1 = crossover_blx(p1, p2, rng)
                c2 = crossover_blx(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Мутация + обрезка
            c1 = repair(mutate(c1, rng, sigma))
            c2 = repair(mutate(c2, rng, sigma))

            f1 = schwefel(c1)
            f2 = schwefel(c2)
            n_evals += 2

            # Для адаптивной мутации считаем "успехи"
            if adaptive:
                n_success += int(f1 < prev_mean) + int(f2 < prev_mean)
                n_trials += 2

            new_pop.append(c1); new_fit.append(f1)
            if len(new_pop) < pop_size:
                new_pop.append(c2); new_fit.append(f2)

        pop = np.array(new_pop)
        fit = np.array(new_fit)

        # Правило 1/5: если >20% успешны — увеличить σ, иначе уменьшить
        if adaptive and n_trials > 0:
            rate = n_success / n_trials
            if rate > 0.2:
                sigma = min(sigma * 1.1, HIGH - LOW)
            elif rate < 0.2:
                sigma = max(sigma * 0.9, 1e-9)

        # Запоминаем, если нашли что-то лучше
        i = int(np.argmin(fit))
        if fit[i] < best_f:
            best_f = float(fit[i])
            best_x = pop[i].copy()

    history.append({
        "gen": generations,
        "best": float(fit.min()),
        "mean": float(fit.mean()),
        "std": float(fit.std()),
    })
    return best_x, best_f, history, n_evals


# ---------------------- ЭКСПЕРИМЕНТЫ ----------------------
def experiment_baseline(n_runs=20):
    """20 независимых запусков базовой конфигурации."""
    print(f"Эксперимент 1: {n_runs} независимых запусков ГА")
    rows, histories = [], []
    for seed in range(n_runs):
        x, f, hist, ne = run_ga(seed)
        rows.append({
            "run": seed, "seed": seed, "best_f": f,
            "n_evals": ne,
            "dist_to_opt": float(np.linalg.norm(x - OPT_X)),
        })
        histories.append(hist)
        print(f"  запуск {seed:2d}: f = {f:.6f}")
    return pd.DataFrame(rows), histories


def print_stats(df):
    print("\nСтатистика по 20 запускам:")
    print(f"  Лучший   : {df.best_f.min():.6f}")
    print(f"  Средний  : {df.best_f.mean():.6f}")
    print(f"  Медиана  : {df.best_f.median():.6f}")
    print(f"  СКО      : {df.best_f.std():.6f}")
    print(f"  Худший   : {df.best_f.max():.6f}")
    print(f"  Успешных (f<1): {(df.best_f < 1).mean():.2%}")


def plot_convergence(histories):
    """График сходимости: среднее по 20 запускам + полоса min-max."""
    bests = np.array([[h["best"] for h in hist] for hist in histories])
    means = np.array([[h["mean"] for h in hist] for hist in histories])
    gens = np.array([h["gen"] for h in histories[0]])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(gens, bests.mean(axis=0), label="среднее лучших")
    ax[0].fill_between(gens, bests.min(axis=0), bests.max(axis=0),
                       alpha=0.2, label="min-max")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Поколение"); ax[0].set_ylabel("Лучший фитнес (log)")
    ax[0].set_title("Лучший фитнес по поколениям")
    ax[0].grid(True, alpha=0.3); ax[0].legend()

    ax[1].plot(gens, means.mean(axis=0), color="orange")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Поколение"); ax[1].set_ylabel("Средний фитнес (log)")
    ax[1].set_title("Средний фитнес популяции")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/convergence.png", dpi=150)
    plt.close(fig)


def experiment_pop_size(n_runs=20):
    """Сравнение pop_size при фиксированном бюджете FES."""
    print("\nЭксперимент 2: сравнение размера популяции (бюджет фиксирован)")
    budget = POP_SIZE * GENERATIONS
    rows = []
    for ps in [30, 60, 100]:
        gens = budget // ps
        fs = []
        for seed in range(n_runs):
            _, f, _, _ = run_ga(seed, pop_size=ps, generations=gens)
            fs.append(f)
        rows.append({
            "pop_size": ps, "generations": gens, "budget": budget,
            "best": min(fs), "mean": float(np.mean(fs)),
            "median": float(np.median(fs)), "std": float(np.std(fs)),
        })
        print(f"  pop={ps:3d}, gen={gens:4d}: медиана f = {np.median(fs):.4f}")
    return pd.DataFrame(rows)


def experiment_ga_vs_random(n_runs=20):
    """Сравнение ГА и случайного поиска при том же бюджете FES."""
    print("\nЭксперимент 3: ГА против случайного поиска")
    budget = POP_SIZE * GENERATIONS
    rs_results = []
    for seed in range(n_runs):
        rng = np.random.default_rng(seed + 10_000)
        best = np.inf
        remaining = budget
        while remaining > 0:
            batch = min(1000, remaining)
            pts = rng.uniform(LOW, HIGH, size=(batch, D))
            for p in pts:
                f = schwefel(p)
                if f < best:
                    best = f
            remaining -= batch
        rs_results.append(best)
    print(f"  Случайный поиск: медиана f = {np.median(rs_results):.4f}")
    return pd.DataFrame({"method": ["RandomSearch"] * n_runs, "best_f": rs_results})


def experiment_adaptive(n_runs=20):
    """Доп. задание: адаптивная мутация (правило 1/5) против фиксированной."""
    print("\nДоп. задание: адаптивная мутация (правило 1/5)")
    rows = []
    for label, adap in [("fixed", False), ("adaptive", True)]:
        fs = []
        for seed in range(n_runs):
            _, f, _, _ = run_ga(seed, adaptive=adap)
            fs.append(f)
        rows.append({"config": label,
                     "best": min(fs), "mean": float(np.mean(fs)),
                     "median": float(np.median(fs)), "std": float(np.std(fs))})
        print(f"  {label:10s}: медиана f = {np.median(fs):.4f}")
    return pd.DataFrame(rows)


# ---------------------- ГЛАВНАЯ ФУНКЦИЯ ----------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    df_runs, histories = experiment_baseline(20)
    df_runs.to_csv("results/runs.csv", index=False)
    print_stats(df_runs)
    plot_convergence(histories)

    df_pop = experiment_pop_size(20)
    df_pop.to_csv("results/pop_comparison.csv", index=False)

    df_rand = experiment_ga_vs_random(20)
    df_ga_vs = pd.concat([
        df_runs[["best_f"]].assign(method="GA"),
        df_rand,
    ], ignore_index=True)
    df_ga_vs.to_csv("results/ga_vs_random.csv", index=False)

    df_adap = experiment_adaptive(20)
    df_adap.to_csv("results/adaptive.csv", index=False)

    print("\nВсе результаты в папке results/")