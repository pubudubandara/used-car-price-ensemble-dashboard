import json
import numpy as np, pandas as pd, joblib, streamlit as st
import xgboost

st.set_page_config(page_title="Sri Lankan Used-Car Prices | Ensemble Dashboard", page_icon="🚗", layout="wide")
LAKH = 100_000

@st.cache_resource
def load_models():
    return joblib.load("artifacts/top_models.joblib")

@st.cache_data
def load_tables():
    meta = json.load(open("artifacts/meta.json", encoding="utf-8"))
    return (pd.read_csv("artifacts/clean_data.csv"), pd.read_csv("artifacts/results.csv"),
        pd.read_csv("artifacts/feature_importance.csv"), pd.read_csv("artifacts/test_predictions.csv"),
        meta, open("artifacts/interpretation.md", encoding="utf-8").read())
models = load_models()
df, res, imp, tp, meta, interp = load_tables()
CAT, NUM = meta["CAT"], meta["NUM"]

st.title("🚗 Sri Lankan Used-Car Price Prediction")
st.caption("CM4371 Ensemble Learning Dashboard  |  Bagging vs Boosting vs Stacking vs Hybrid  |  "
           "Data: ikman.lk & Riyasewana.com (Kaggle, Prasad Nirmal)  |  Prices in lakhs (1 lakh = LKR 100,000)")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🏆 Model Comparison", "💰 Predict a Price"])

with tab1:
    a, b, c, d = st.columns(4)
    a.metric("Listings used", f"{len(df):,}")
    b.metric("Brands", df["Brand"].nunique())
    c.metric("Median price", f"{df['Price'].median():.0f} lakhs")
    d.metric("Best model", meta["best"])
    left, right = st.columns(2)
    top_brands = df["Brand"].value_counts().head(12).index
    left.subheader("Median price by brand (12 most advertised)")
    left.bar_chart(df[df["Brand"].isin(top_brands)].groupby("Brand")["Price"].median().sort_values(ascending=False))
    right.subheader("Median price by car age (years)")
    right.line_chart(df.groupby("Age")["Price"].median())
    st.subheader("Median price by fuel type")
    st.bar_chart(df.groupby("Fuel Type")["Price"].median())

with tab2:
    st.subheader("All models on the same held-out test set")
    st.dataframe(res.style.format({"MAE (lakhs)": "{:.2f}", "RMSE (lakhs)": "{:.2f}", "MAPE (%)": "{:.1f}",
                                   "R2 (log)": "{:.3f}", "Fit time (s)": "{:.1f}"}))
    c1, c2 = st.columns(2)
    c1.subheader("MAE per model (lakhs, lower = better)")
    c1.bar_chart(res.set_index("Model")["MAE (lakhs)"])
    c2.subheader("MAE per family (best vs average)")
    fam = res.groupby("Family")["MAE (lakhs)"].agg(["min", "mean"])
    c2.bar_chart(fam)
    d1, d2 = st.columns(2)
    d1.subheader("Predicted vs actual")
    pick = d1.selectbox("Model", [c for c in tp.columns if c != "actual"], index=0)
    d1.scatter_chart(pd.DataFrame({"Actual (lakhs)": tp["actual"], "Predicted (lakhs)": tp[pick]}),
                     x="Actual (lakhs)", y="Predicted (lakhs)")
    d2.subheader(f"What drives the price ({meta['best']})")
    d2.bar_chart(imp.set_index("Feature")["Importance"])
    st.subheader("Interpretation: Analysis of Model Performance")
    st.markdown(interp)

with tab3:
    st.subheader("Describe the car")
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", sorted(df["Brand"].unique()), index=sorted(df["Brand"].unique()).index("TOYOTA") if "TOYOTA" in set(df["Brand"]) else 0)
    model_name = c2.selectbox("Model", sorted(df[df["Brand"] == brand]["Model"].unique()))
    yom = c3.slider("Year of manufacture", int(2025 - df["Age"].max()), 2025, 2016)
    same = df[(df["Brand"] == brand) & (df["Model"] == model_name)]
    cc_default = int(same["Engine (cc)"].median()) if len(same) else int(df["Engine (cc)"].median())
    c4, c5, c6 = st.columns(3)
    cc = c4.number_input("Engine capacity (cc)", 300, 6000, cc_default, step=50, key=f"cc_{brand}_{model_name}")
    gear = c5.selectbox("Gear", sorted(df["Gear"].unique()))
    fuel = c6.selectbox("Fuel type", sorted(df["Fuel Type"].unique()))
    c7, c8, c9 = st.columns(3)
    town = c7.selectbox("Town", sorted(df["Town"].unique()))
    cond = c8.selectbox("Condition", sorted(df["Condition"].unique()))
    lease = c9.selectbox("Leasing", sorted(df["Leasing"].unique()))
    e1, e2, e3, e4 = st.columns(4)
    ac = e1.selectbox("Air condition", sorted(df["AIR CONDITION"].unique()))
    ps = e2.selectbox("Power steering", sorted(df["POWER STEERING"].unique()))
    pm = e3.selectbox("Power mirror", sorted(df["POWER MIRROR"].unique()))
    pw = e4.selectbox("Power window", sorted(df["POWER WINDOW"].unique()))

    if st.button("Predict price", type="primary"):
        row = {"Brand": brand, "Model": model_name, "Town": town, "Gear": gear, "Fuel Type": fuel, "Leasing": lease,
               "Condition": cond, "AIR CONDITION": ac, "POWER STEERING": ps, "POWER MIRROR": pm, "POWER WINDOW": pw,
               "Age": meta["ref_year"] - yom, "Engine (cc)": cc}
        X_new = pd.DataFrame([row])[CAT + NUM]
        out = {n: float(np.expm1(m.predict(X_new))[0]) for n, m in models.items()}
        avg = float(np.mean(list(out.values())))
        st.success(f"Average of the top models: **{avg:,.1f} lakhs  (about LKR {avg * LAKH:,.0f})**")
        st.table(pd.DataFrame({"Model": list(out.keys()),
                               "Predicted (lakhs)": [f"{v:,.1f}" for v in out.values()],
                               "Predicted (LKR)": [f"{v * LAKH:,.0f}" for v in out.values()]}))
        st.caption("Estimates reflect advertised prices (Dec 2024 - Feb 2025), not final selling prices, and typically "
                   "carry an error of the size shown in the Model Comparison tab.")
