# Sri Lankan Used-Car Price Prediction Dashboard

** Machine Learning & Pattern Recognition – Hands-on Assignment**  
**Live Dashboard:** [Sri Lankan Used-Car Price Prediction App](https://used-car-price-ensemble-dashboard.streamlit.app/)

Streamlit dashboard for comparing ensemble regressors that predict used-car prices in Sri Lanka across **Bagging**, **Boosting**, **Stacking**, and **Hybrid** paradigms. Prices are reported in lakhs (1 lakh = LKR 100,000).

## Project contents

- `Car_Price_Dashboard.ipynb` - full exploratory data analysis, model training, evaluation, validation checks, and artifact generation
- `train.py` - reproducible local retraining and interpretation generation script
- `app.py` - interactive Streamlit dashboard
- `requirements.txt` - Python dependencies for local execution and Streamlit Cloud deployment
- `artifacts/clean_data.csv` - cleaned dataset used for dashboard visualizations
- `artifacts/results.csv` - held-out test set evaluation metrics across all 5 models
- `artifacts/feature_importance.csv` - permutation importance scores for the best model
- `artifacts/test_predictions.csv` - actual vs. predicted values on the held-out test set
- `artifacts/interpretation.md` - shared model-performance analysis
- `artifacts/meta.json` - feature metadata and top model registry
- `artifacts/top_models.joblib` - serialized top three models for live inference

## Dataset

The source data is the Kaggle **SriLankan Vehicle (Car) Price Dataset** (Prasad Nirmal), scraped from **ikman.lk** and **Riyasewana.com** (Dec 2024 – Feb 2025). The preprocessing workflow drops the `Date` and `Millage(KM)` columns, removes duplicate listings prior to splitting, derives vehicle `Age` from `YOM`, groups rare models and towns into `OTHER`, applies a `log1p` target transformation, and evaluates on an 80/20 held-out split (`random_state=42`).

## Run the training script

From the project root directory:

```powershell
python train.py
