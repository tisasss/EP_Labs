import pandas as pd

print("=" * 60)
print("results/runs.csv")
print("=" * 60)
df = pd.read_csv("results/runs.csv")
print(f"Строк: {len(df)}   (должно быть 20)")
print(f"Столбцы: {list(df.columns)}")
print(f"best_f:    min={df.best_f.min():.6f}  max={df.best_f.max():.6f}  "
      f"mean={df.best_f.mean():.4f}  median={df.best_f.median():.4f}")
print(f"n_evals:   min={df.n_evals.min()}   max={df.n_evals.max()}   (должно быть 17460)")
print(f"dist_to_opt: min={df.dist_to_opt.min():.4f}  max={df.dist_to_opt.max():.4f}")
print()
print(df[["run", "best_f", "n_evals", "dist_to_opt"]].to_string(index=False))

for name in ["pop_comparison", "ga_vs_random", "adaptive"]:
    print()
    print("=" * 60)
    print(f"results/{name}.csv")
    print("=" * 60)
    print(pd.read_csv(f"results/{name}.csv").to_string(index=False))