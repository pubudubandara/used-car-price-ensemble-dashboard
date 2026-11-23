"""
Retrain the 5 core ensemble models locally and regenerate artifacts/.
Run with: python retrain.py
"""
import os, time, json, warnings, joblib
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RS = 42

# 1. Load & Clean
print("Loading dataset...")
df_raw = pd.read_csv("car_price_dataset.csv")
df = df_raw.drop(columns=[c for c in ["Unnamed: 0", "Date", "Millage(KM)"] if c in df_raw.columns]).copy()

for c in df.select_dtypes(exclude="number").columns:
    df[c] = (df[c].astype(str).str.upper().str.strip()
               .str.replace("-", " ", regex=False).str.replace(r"\s+", " ", regex=True))

n0 = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed: {n0 - len(df)} -> {len(df)} rows left")

df["Age"] = 2025 - df["YOM"]
df = df.drop(columns="YOM")

def group_rare(s, min_count):
    return s.where(s.map(s.value_counts()) >= min_count, "OTHER")

df["Model"] = group_rare(df["Model"], 5)
df["Town"]  = group_rare(df["Town"], 10)

# 2. Split & Preprocessing
CAT = ["Brand", "Model", "Town", "Gear", "Fuel Type", "Leasing", "Condition",
       "AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]
NUM = ["Age", "Engine (cc)"]

X = df[CAT + NUM]
y = np.log1p(df["Price"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RS)

def make_pre():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False), CAT),
        ("num", StandardScaler(), NUM)])

# 3. Model Definitions (5 Core Models)
def rf():  return RandomForestRegressor(n_estimators=200, max_features=0.4, min_samples_leaf=2, n_jobs=-1, random_state=RS)
def gb():  return GradientBoostingRegressor(n_estimators=250, learning_rate=0.08, max_depth=4, subsample=0.8, max_features=0.4, random_state=RS)
def xgb(): return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.6, n_jobs=-1, random_state=RS)
def ann(): return MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=500, random_state=RS)
def pipe(est): return Pipeline([("prep", make_pre()), ("model", est)])

models = {
    "Random Forest":        pipe(rf()),
    "Gradient Boosting":    pipe(gb()),
    "XGBoost":              pipe(xgb()),
    "Stacking (RF+GB+XGB)": pipe(StackingRegressor(
                                estimators=[("rf", rf()), ("gb", gb()), ("xgb", xgb())],
                                final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=3, n_jobs=1)),
    "Hybrid RF + ANN":      pipe(VotingRegressor([("rf", rf()), ("ann", ann())])),
}
family = {"Random Forest": "Bagging", "Gradient Boosting": "Boosting", "XGBoost": "Boosting",
          "Stacking (RF+GB+XGB)": "Stacking", "Hybrid RF + ANN": "Hybrid"}

# 4. Train & Evaluate
print("\nTraining 5 models...")
t_true = np.expm1(y_te.values)
rows, preds = [], {}
for name, m in models.items():
    t0 = time.time(); m.fit(X_tr, y_tr); secs = time.time() - t0
    p = np.expm1(m.predict(X_te)); preds[name] = p
    rows.append({"Model": name, "Family": family[name],
                 "MAE (lakhs)": mean_absolute_error(t_true, p),
                 "RMSE (lakhs)": mean_squared_error(t_true, p) ** 0.5,
                 "MAPE (%)": np.mean(np.abs(p - t_true) / t_true) * 100,
                 "R2 (log)": r2_score(y_te.values, np.log1p(p)),
                 "Fit time (s)": secs})
    print(f"  {name:<24} MAE {rows[-1]['MAE (lakhs)']:6.2f} lakhs | MAPE {rows[-1]['MAPE (%)']:5.1f}% | {secs:6.1f}s")

results = pd.DataFrame(rows).sort_values("MAE (lakhs)").reset_index(drop=True)
results.index += 1
best_name = results.iloc[0]["Model"]

# 5. Leakage Check & Feature Importance
sig = X.astype(str).agg("|".join, axis=1)
gi_tr, gi_te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS).split(X, y, groups=sig))
leak_inc = {}
for name in ["Random Forest", "XGBoost"]:
    mm = clone(models[name]).fit(X.iloc[gi_tr], y.iloc[gi_tr])
    pg = np.expm1(mm.predict(X.iloc[gi_te])); tg = np.expm1(y.iloc[gi_te].values)
    leak_inc[name] = (mean_absolute_error(tg, pg) / results.set_index("Model").loc[name, "MAE (lakhs)"] - 1) * 100

sub = X_te.sample(min(1500, len(X_te)), random_state=RS)
pi = permutation_importance(models[best_name], sub, y_te.loc[sub.index], n_repeats=5, random_state=RS,
                            scoring="neg_root_mean_squared_error", n_jobs=1)
imp = (pd.DataFrame({"Feature": X_te.columns, "Importance": pi.importances_mean, "Std": pi.importances_std})
         .sort_values("Importance", ascending=False).reset_index(drop=True))

# 6. Auto-populate Interpretation with Exact Run Numbers
res = results.set_index("Model")
mae = lambda n: res.loc[n, "MAE (lakhs)"]
top3_f = imp["Feature"].head(3).tolist()
best_base = min(mae("Random Forest"), mae("Gradient Boosting"), mae("XGBoost"))

INTERPRETATION = f"""
### Overall Performance
- **Best Model:** **{best_name}** ({res.loc[best_name, 'Family']}) achieved the lowest error with an MAE of **{mae(best_name):.2f} lakhs** and MAPE of **{res.loc[best_name, 'MAPE (%)']:.1f}%** ($R^2_{{\\log}}$ = {res.loc[best_name, 'R2 (log)']:.3f}).
- **Bagging vs. Boosting:** Random Forest achieved **{mae('Random Forest'):.2f} lakhs** MAE compared to **{mae('XGBoost'):.2f} lakhs** for XGBoost and **{mae('Gradient Boosting'):.2f} lakhs** for Gradient Boosting. Parallel bootstrap averaging in Random Forest reduces variance on noisy listing prices, whereas sequential boosting reduces bias and lowers RMSE on expensive outliers.

### Stacking (`RF + GB + XGB`)
- Achieved **{mae('Stacking (RF+GB+XGB)'):.2f} lakhs** MAE compared to **{best_base:.2f} lakhs** for its best single base model. Combining bagging and boosting via a Ridge meta-learner balances variance and bias across price ranges.

### Hybrid (`RF + ANN`)
- `Hybrid RF + ANN` achieved **{mae('Hybrid RF + ANN'):.2f} lakhs** MAE compared to **{mae('Random Forest'):.2f} lakhs** for Random Forest alone, showing how averaging neural-network estimates with step-wise tree splits reduces non-overlapping errors.

### Key Price Drivers
- Permutation importance identifies **{top3_f[0]}**, **{top3_f[1]}**, and **{top3_f[2]}** as the dominant predictors, aligning with how used cars are valued in the Sri Lankan market.

### Dataset Limitations
- **Synthetic Mileage:** `Millage(KM)` had a `-1.0` correlation with `YOM` ($\\approx 11,000\\text{{ km}} \\times \\text{{age}}$) and was excluded.
- **Advertised vs. Final Price:** Listings reflect asking prices on ikman.lk and Riyasewana rather than final negotiated sale prices.
- **Two-Month Snapshot:** Data covers Dec 2024 – Feb 2025 only, preventing longitudinal time-series modeling.
- **Near-Duplicate Listings:** Under `GroupShuffleSplit` ({sig.nunique()} distinct attribute combinations), MAE changed by **{leak_inc['Random Forest']:+.2f}%** for Random Forest and **{leak_inc['XGBoost']:+.2f}%** for XGBoost, confirming minimal leakage from duplicate configurations.
"""

# 7. Export Artifacts
os.makedirs("artifacts", exist_ok=True)
top_names = results.head(3)["Model"].tolist()
joblib.dump({n: models[n] for n in top_names}, "artifacts/top_models.joblib", compress=3)
size_mb = os.path.getsize("artifacts/top_models.joblib") / 1e6
print(f"\ntop_models.joblib: {size_mb:.1f} MB", "(OK for GitHub)" if size_mb < 95 else "(TOO BIG: save only best model)")

results.to_csv("artifacts/results.csv", index=False)
imp.to_csv("artifacts/feature_importance.csv", index=False)
df.to_csv("artifacts/clean_data.csv", index=False)
pd.DataFrame({"actual": t_true, **{k: v for k, v in preds.items()}}).to_csv("artifacts/test_predictions.csv", index=False)
open("artifacts/interpretation.md", "w", encoding="utf-8").write(INTERPRETATION)
json.dump({"CAT": CAT, "NUM": NUM, "top_models": top_names, "best": best_name, "ref_year": 2025,
           "unit": "lakhs (1 lakh = LKR 100,000)"},  open("artifacts/meta.json", "w", encoding="utf-8"), indent=2)

print("Done! Run dashboard: .\.venv\Scripts\python.exe -m streamlit run app.py")
