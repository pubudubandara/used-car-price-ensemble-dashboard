
### Overall Performance
- **Best Model:** **Stacking (RF+GB+XGB)** (Stacking) achieved the lowest error with an MAE of **5.70 lakhs** and MAPE of **13.2%** ($R^2_{\log}$ = 0.921).
- **Bagging vs. Boosting:** Random Forest achieved **5.80 lakhs** MAE compared to **5.94 lakhs** for XGBoost and **7.11 lakhs** for Gradient Boosting. Parallel bootstrap averaging in Random Forest reduces variance on noisy listing prices, whereas sequential boosting reduces bias and lowers RMSE on expensive outliers.

### Stacking (`RF + GB + XGB`)
- Achieved **5.70 lakhs** MAE compared to **5.80 lakhs** for its best single base model. Combining bagging and boosting via a Ridge meta-learner balances variance and bias across price ranges.

### Hybrid (`RF + ANN`)
- `Hybrid RF + ANN` achieved **5.71 lakhs** MAE compared to **5.80 lakhs** for Random Forest alone, showing how averaging neural-network estimates with step-wise tree splits reduces non-overlapping errors.

### Key Price Drivers
- Permutation importance identifies **Age**, **Brand**, and **Gear** as the dominant predictors, aligning with how used cars are valued in the Sri Lankan market.

### Dataset Limitations
- **Synthetic Mileage:** `Millage(KM)` had a `-1.0` correlation with `YOM` ($\approx 11,000\text{ km} \times \text{age}$) and was excluded.
- **Advertised vs. Final Price:** Listings reflect asking prices on ikman.lk and Riyasewana rather than final negotiated sale prices.
- **Two-Month Snapshot:** Data covers Dec 2024 – Feb 2025 only, preventing longitudinal time-series modeling.
- **Near-Duplicate Listings:** Under `GroupShuffleSplit` (8842 distinct attribute combinations), MAE changed by **-1.32%** for Random Forest and **+1.21%** for XGBoost, confirming minimal leakage from duplicate configurations.
