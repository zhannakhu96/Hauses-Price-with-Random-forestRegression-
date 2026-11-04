import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_absolute_error

RS = 42
rng = np.random.default_rng(RS)
n = 2000
df = pd.DataFrame({
    "living_area_m2": rng.normal(120, 40, n).clip(40, 350),
    "plot_area_m2": rng.normal(450, 200, n).clip(80, 1500),
    "rooms": rng.integers(2, 9, n),
    "year_built": rng.integers(1900, 2024, n),
    "dist_center_km": rng.exponential(8, n).clip(0.5, 40),
    "energy_class": rng.integers(1, 9, n),
})
df["price"] = (2800*df.living_area_m2 + 120*df.plot_area_m2 + 9000*df.rooms
    + 1500*(df.year_built-1900) - 7000*df.dist_center_km - 6000*df.energy_class
    + rng.normal(0, 40000, n)).clip(lower=80000).round(-2)
df.to_csv("haus_synthetic.csv", index=False)

X, y = df.drop(columns="price").astype(float), df.price
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5,
                           max_depth=20, random_state=RS, n_jobs=-1).fit(Xtr, ytr)
p = rf.predict(Xte); res = yte - p
print(f"R2={r2_score(yte,p):.4f} MAE={mean_absolute_error(yte,p):,.0f}")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})

# 1 actual vs predicted
fig, ax = plt.subplots(figsize=(6,5.5))
ax.scatter(yte/1e3, p/1e3, s=10, alpha=.5, color="#2a6f97")
lim = [yte.min()/1e3, yte.max()/1e3]; ax.plot(lim, lim, "r--", lw=1)
ax.set(xlabel="Actual price (k€)", ylabel="Predicted price (k€)",
       title=f"Actual vs predicted (R²={r2_score(yte,p):.3f})")
fig.tight_layout(); fig.savefig("plots/1_actual_vs_pred.png"); plt.close()

# 2 residuals
fig, (a, b) = plt.subplots(1, 2, figsize=(11,4.5))
a.hist(res/1e3, bins=40, color="#468faf"); a.axvline(0, color="r", ls="--")
a.set(title="Residual distribution", xlabel="Residual (k€)", ylabel="Count")
b.scatter(p/1e3, res/1e3, s=10, alpha=.5, color="#2a6f97"); b.axhline(0, color="r", ls="--")
b.set(title="Residuals vs predicted", xlabel="Predicted (k€)", ylabel="Residual (k€)")
fig.tight_layout(); fig.savefig("plots/2_residuals.png"); plt.close()

# 3 importance
pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=RS, n_jobs=-1)
imp = pd.Series(pi.importances_mean, X.columns).sort_values()
err = pd.Series(pi.importances_std, X.columns)[imp.index]
fig, ax = plt.subplots(figsize=(7,4.5))
ax.barh(imp.index, imp.values, xerr=err.values, color="#2a6f97")
ax.set(title="Permutation importance (drop in R²)", xlabel="Mean importance")
fig.tight_layout(); fig.savefig("plots/3_importance.png"); plt.close()

# 4 partial dependence
fig, ax = plt.subplots(figsize=(11,7))
PartialDependenceDisplay.from_estimator(rf, Xte, ["living_area_m2","year_built","dist_center_km","plot_area_m2"], ax=ax)
fig.suptitle("Partial dependence (price in €)"); fig.tight_layout(); fig.savefig("plots/4_partial_dependence.png"); plt.close()

# 5 error by price band
q = pd.qcut(yte, 5); d = pd.DataFrame({"band": q, "res": res, "ape": np.abs(res/yte)*100})
g = d.groupby("band", observed=True).agg(bias=("res","mean"), mape=("ape","mean"))
print(g.round(1))
fig, (a, b) = plt.subplots(1, 2, figsize=(11,4.5))
lbl = [f"Q{i+1}" for i in range(5)]
a.bar(lbl, g.bias/1e3, color=["#c1121f" if v<0 else "#2a6f97" for v in g.bias]); a.axhline(0,color="k",lw=.8)
a.set(title="Mean residual by price quintile", ylabel="k€")
b.bar(lbl, g.mape, color="#468faf"); b.set(title="MAPE by price quintile", ylabel="%")
fig.tight_layout(); fig.savefig("plots/5_error_by_price_band.png"); plt.close()

# 6 correlation
fig, ax = plt.subplots(figsize=(6.5,5.5))
c = df.corr(); im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns); ax.grid(False)
for i in range(len(c)):
    for j in range(len(c)): ax.text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im); ax.set_title("Correlation matrix"); fig.tight_layout()
fig.savefig("plots/6_correlation.png"); plt.close()
print(df.price.describe().round(0))
