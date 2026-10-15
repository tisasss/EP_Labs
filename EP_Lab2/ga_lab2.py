
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N = 35
BUDGET_RATIO = 0.70
VOLUME_RATIO = 0.55

POP_SIZE = 60
GENERATIONS = 200
ELITE = 2
PC = 0.9
PM = 0.2
SHIFT_MAX = 3


def generate_data(seed=0):
    rng = np.random.default_rng(seed)
    cost = rng.integers(20, 300, size=N).astype(int)
    demand = rng.integers(10, 50, size=N).astype(int)
    min_stock = (0.3 * demand).astype(int)
    weight = rng.integers(1, 6, size=N)
    volume = rng.integers(1, 20, size=N)
    budget = int(np.sum(cost * demand) * BUDGET_RATIO)
    volume_cap = int(np.sum(volume * demand) * VOLUME_RATIO)
    return {
        "cost": cost, "demand": demand, "min_stock": min_stock,
        "weight": weight, "volume": volume,
        "budget": budget, "volume_cap": volume_cap,
    }


def total_cost(x, cost):
    return int(np.sum(cost * x))


def total_volume(x, volume):
    return int(np.sum(volume * x))


def deficit(x, demand, weight):
    return float(np.sum(weight * np.maximum(0, demand - x)))


def is_feasible(x, data):
    if np.any(x < data["min_stock"]):
        return False
    if np.any(x > data["demand"]):
        return False
    if total_cost(x, data["cost"]) > data["budget"]:
        return False
    if total_volume(x, data["volume"]) > data["volume_cap"]:
        return False
    return True


def fitness_penalty(x, data, lam=1.0):
    d = deficit(x, data["demand"], data["weight"])
    over_c = max(0, total_cost(x, data["cost"]) - data["budget"])
    over_v = max(0, total_volume(x, data["volume"]) - data["volume_cap"])
    max_def = float(np.sum(data["weight"] * data["demand"]))
    penalty = lam * (over_c / data["budget"] + over_v / data["volume_cap"]) * max_def
    return d + penalty


def fitness_repair(x, data):
    x_rep = repair_constraints(x, data)
    return deficit(x_rep, data["demand"], data["weight"])


def repair_constraints(x, data):
    x = np.clip(x, data["min_stock"], data["demand"]).copy()
    score = data["weight"] / (data["cost"] + data["volume"])
    order = np.argsort(score)

    for i in order:
        total_c = int(np.sum(data["cost"] * x))
        total_v = int(np.sum(data["volume"] * x))
        if total_c <= data["budget"] and total_v <= data["volume_cap"]:
            break
        over_c = max(0, total_c - data["budget"])
        over_v = max(0, total_v - data["volume_cap"])
        need_c = int(np.ceil(over_c / data["cost"][i])) if data["cost"][i] > 0 else 0
        need_v = int(np.ceil(over_v / data["volume"][i])) if data["volume"][i] > 0 else 0
        reduce_by = min(max(need_c, need_v), int(x[i] - data["min_stock"][i]))
        x[i] -= reduce_by
    return x


def init_population(pop_size, data, rng):
    pop = np.zeros((pop_size, N), dtype=int)
    for k in range(pop_size):
        x = rng.integers(data["min_stock"], data["demand"] + 1)
        pop[k] = repair_constraints(x, data)
    return pop


def tournament(pop, fit, rng, k=3):
    idx = rng.integers(0, len(pop), k)
    return pop[idx[np.argmin(fit[idx])]].copy()


def crossover_uniform(p1, p2, rng):
    mask = rng.random(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2


def mutation_simple(x, data, rng):
    x = x.copy()
    mask = rng.random(len(x)) < PM
    steps = rng.integers(-SHIFT_MAX, SHIFT_MAX + 1, size=len(x))
    x = x + mask * steps
    return np.clip(x, data["min_stock"], data["demand"])


def mutation_shift(x, data, rng):
    x = x.copy()
    for _ in range(3):
        i = int(rng.integers(len(x)))
        if x[i] < data["demand"][i]:
            x[i] += 1
            j = int(rng.integers(len(x)))
            while j == i:
                j = int(rng.integers(len(x)))
            if x[j] > data["min_stock"][j]:
                x[j] -= 1
    return np.clip(x, data["min_stock"], data["demand"])


def run_ga(seed, data, method="repair", mutation="simple",
           pop_size=POP_SIZE, generations=GENERATIONS):
    rng = np.random.default_rng(seed)

    if method == "repair":
        def fit_func(x):
            return fitness_repair(x, data)
    else:
        def fit_func(x):
            return fitness_penalty(x, data, lam=1.0)

    pop = init_population(pop_size, data, rng)
    fit = np.array([fit_func(ind) for ind in pop])
    n_evals = pop_size

    best_x, best_f = None, np.inf
    best_gen = 0
    for i in range(len(pop)):
        if is_feasible(pop[i], data) and fit[i] < best_f:
            best_f = float(fit[i])
            best_x = pop[i].copy()

    history = []

    for gen in range(generations):
        history.append({"gen": gen,
                        "best": float(fit.min()),
                        "mean": float(fit.mean())})

        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:ELITE]]
        new_fit = [float(fit[i]) for i in order[:ELITE]]

        while len(new_pop) < pop_size:
            p1 = tournament(pop, fit, rng)
            p2 = tournament(pop, fit, rng)

            if rng.random() < PC:
                c1, c2 = crossover_uniform(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if mutation == "simple":
                c1 = mutation_simple(c1, data, rng)
                c2 = mutation_simple(c2, data, rng)
            else:
                c1 = mutation_shift(c1, data, rng)
                c2 = mutation_shift(c2, data, rng)

            if method == "repair":
                c1 = repair_constraints(c1, data)
                c2 = repair_constraints(c2, data)
            else:
                c1 = np.clip(c1, data["min_stock"], data["demand"])
                c2 = np.clip(c2, data["min_stock"], data["demand"])

            f1 = fit_func(c1)
            f2 = fit_func(c2)
            n_evals += 2

            new_pop.append(c1); new_fit.append(f1)
            if len(new_pop) < pop_size:
                new_pop.append(c2); new_fit.append(f2)

        pop = np.array(new_pop)
        fit = np.array(new_fit)

        for i in range(len(pop)):
            if is_feasible(pop[i], data) and fit[i] < best_f:
                best_f = float(fit[i])
                best_x = pop[i].copy()
                best_gen = gen

    history.append({"gen": generations,
                    "best": float(fit.min()),
                    "mean": float(fit.mean())})
    return best_x, best_f, history, n_evals, best_gen


def greedy_solution(data):
    x = data["min_stock"].copy()
    remaining = data["budget"] - total_cost(x, data["cost"])
    order = np.argsort(-data["weight"] / data["cost"])
    for i in order:
        can_buy = int(data["demand"][i] - x[i])
        can_afford = int(remaining // data["cost"][i])
        buy = min(can_buy, can_afford)
        x[i] += buy
        remaining -= buy * data["cost"][i]
    return repair_constraints(x, data)


def random_feasible_search(budget, data, seed):
    rng = np.random.default_rng(seed)
    best_x, best_f = None, np.inf
    n = 0
    while n < budget:
        batch = min(500, budget - n)
        for _ in range(batch):
            x = rng.integers(data["min_stock"], data["demand"] + 1)
            x = repair_constraints(x, data)
            f = deficit(x, data["demand"], data["weight"])
            if is_feasible(x, data) and f < best_f:
                best_f, best_x = float(f), x.copy()
        n += batch
    return best_x, best_f


def experiment_baseline(data, n_runs=20, threshold=None):
    print(f"Бюджет: {data['budget']}, объём: {data['volume_cap']}")
    print(f"Полная стоимость: {total_cost(data['demand'], data['cost'])}, "
          f"полный объём: {total_volume(data['demand'], data['volume'])}")
    rows, histories = [], []
    for seed in range(n_runs):
        x, f, hist, ne, bgen = run_ga(seed, data, method="repair",
                                      mutation="simple")
        rows.append({
            "run": seed, "seed": seed,
            "best_f": f,
            "cost": total_cost(x, data["cost"]),
            "volume": total_volume(x, data["volume"]),
            "n_evals": ne,
            "best_gen": bgen,
        })
        histories.append(hist)
        print(f"  run {seed:2d}: дефицит = {f:8.2f}, "
              f"стоимость = {rows[-1]['cost']}, "
              f"объём = {rows[-1]['volume']}, "
              f"поколение оптимума = {bgen}")
    df = pd.DataFrame(rows)
    if threshold is not None:
        df.attrs["threshold"] = threshold
    return df, histories


def plot_convergence(histories):
    bests = np.array([[h["best"] for h in hist] for hist in histories])
    means = np.array([[h["mean"] for h in hist] for hist in histories])
    gens = np.array([h["gen"] for h in histories[0]])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(gens, bests.mean(axis=0), label="среднее лучших")
    ax[0].fill_between(gens, bests.min(axis=0), bests.max(axis=0),
                       alpha=0.2, label="min-max")
    ax[0].set_xlabel("Поколение"); ax[0].set_ylabel("Лучший фитнес")
    ax[0].set_title("Лучший фитнес по поколениям")
    ax[0].grid(True, alpha=0.3); ax[0].legend()

    ax[1].plot(gens, means.mean(axis=0), color="orange")
    ax[1].set_xlabel("Поколение"); ax[1].set_ylabel("Средний фитнес")
    ax[1].set_title("Средний фитнес популяции")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/convergence.png", dpi=150)
    plt.close(fig)


def experiment_constraints(data, n_runs=20):
    print("\nСравнение способов обработки ограничений (штраф vs ремонт)")
    rows = []
    for method in ["penalty", "repair"]:
        fs, bgs = [], []
        for seed in range(n_runs):
            _, f, _, _, bgen = run_ga(seed, data, method=method,
                                      mutation="simple")
            fs.append(f); bgs.append(bgen)
        rows.append({
            "method": method,
            "best": float(np.min(fs)),
            "mean": float(np.mean(fs)),
            "median": float(np.median(fs)),
            "std": float(np.std(fs)),
            "mean_best_gen": float(np.mean(bgs)),
        })
        print(f"  {method:8s}: медиана = {np.median(fs):.2f}, "
              f"среднее поколение = {np.mean(bgs):.0f}")
    return pd.DataFrame(rows)


def experiment_mutations(data, n_runs=20):
    print("\nСравнение мутаций (простая vs shift)")
    rows = []
    for mut in ["simple", "shift"]:
        fs, bgs = [], []
        for seed in range(n_runs):
            _, f, _, _, bgen = run_ga(seed, data, method="repair",
                                      mutation=mut)
            fs.append(f); bgs.append(bgen)
        rows.append({
            "mutation": mut,
            "best": float(np.min(fs)),
            "mean": float(np.mean(fs)),
            "median": float(np.median(fs)),
            "std": float(np.std(fs)),
            "mean_best_gen": float(np.mean(bgs)),
        })
        print(f"  {mut:8s}: медиана = {np.median(fs):.2f}, "
              f"среднее поколение = {np.mean(bgs):.0f}")
    return pd.DataFrame(rows)


def experiment_reference(data, n_runs=20):
    print("\nСравнение с жадной эвристикой и случайным поиском")
    x_g = greedy_solution(data)
    f_g = deficit(x_g, data["demand"], data["weight"])
    print(f"  Жадная эвристика: дефицит = {f_g:.2f}")

    budget = POP_SIZE + 2 * POP_SIZE * GENERATIONS
    fs_r = []
    for seed in range(n_runs):
        _, f = random_feasible_search(budget, data, seed + 10_000)
        fs_r.append(f)
    print(f"  Случайный поиск : медиана = {np.median(fs_r):.2f}")

    rows = [{"method": "greedy", "best_f": f_g}]
    for f in fs_r:
        rows.append({"method": "random", "best_f": f})
    return pd.DataFrame(rows), f_g


def save_best_solution(data, x, fname="results/best_solution.csv"):
    df = pd.DataFrame({
        "id": np.arange(N),
        "cost": data["cost"],
        "volume": data["volume"],
        "demand": data["demand"],
        "min_stock": data["min_stock"],
        "weight": data["weight"],
        "buy": x,
        "spent": data["cost"] * x,
        "used_volume": data["volume"] * x,
        "deficit_units": np.maximum(0, data["demand"] - x),
        "deficit_weighted": data["weight"] * np.maximum(0, data["demand"] - x),
    })
    df.to_csv(fname, index=False)


# ---------- main ----------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    data = generate_data(seed=42)
    print(f"N = {N}")
    print(f"Бюджет = {data['budget']}, полная стоимость = "
          f"{total_cost(data['demand'], data['cost'])}")
    print(f"Объём склада = {data['volume_cap']}, полный объём = "
          f"{total_volume(data['demand'], data['volume'])}")

    x_g = greedy_solution(data)
    greedy_deficit = deficit(x_g, data["demand"], data["weight"])
    threshold = 1.2 * greedy_deficit
    print(f"Жадная эвристика: дефицит = {greedy_deficit:.2f}, "
          f"порог успеха = {threshold:.2f}")

    df_runs, histories = experiment_baseline(data, n_runs=20,
                                             threshold=threshold)
    df_runs.to_csv("results/runs.csv", index=False)

    with open("results/summary.json", "w") as f:
        json.dump({
            "greedy_deficit": greedy_deficit,
            "threshold": threshold,
            "budget": int(data["budget"]),
            "full_cost": int(total_cost(data["demand"], data["cost"])),
            "volume_cap": int(data["volume_cap"]),
            "full_volume": int(total_volume(data["demand"], data["volume"])),
        }, f, indent=2)

    print("\nСтатистика по 20 запускам:")
    print(f"  Лучший   : {df_runs.best_f.min():.4f}")
    print(f"  Средний  : {df_runs.best_f.mean():.4f}")
    print(f"  Медиана  : {df_runs.best_f.median():.4f}")
    print(f"  СКО      : {df_runs.best_f.std():.4f}")
    print(f"  Худший   : {df_runs.best_f.max():.4f}")
    print(f"  Доля успешных (< {threshold:.2f}): "
          f"{(df_runs.best_f < threshold).mean():.2%}")
    print(f"  Поколений до оптимума: медиана = "
          f"{df_runs.best_gen.median():.0f}, "
          f"min = {df_runs.best_gen.min()}, max = {df_runs.best_gen.max()}")

    plot_convergence(histories)

    df_c = experiment_constraints(data, n_runs=20)
    df_c.to_csv("results/constraints.csv", index=False)

    df_m = experiment_mutations(data, n_runs=20)
    df_m.to_csv("results/mutations.csv", index=False)

    df_ref, _ = experiment_reference(data, n_runs=20)
    df_ref.to_csv("results/reference.csv", index=False)

    best_row = df_runs.loc[df_runs.best_f.idxmin()]
    best_seed = int(best_row["seed"])
    x_best, f_best, _, _, _ = run_ga(best_seed, data, method="repair", mutation="simple")
    save_best_solution(data, x_best)
    print(f"\nЛучшее решение сохранено, дефицит = {f_best:.2f}")

    print("\nГотово. Результаты в папке results/")