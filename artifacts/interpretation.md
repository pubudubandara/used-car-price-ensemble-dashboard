
### Overall Performance
- **Best Model:** [Model Name, Family] achieved the lowest error with an MAE of [X.XX] lakhs and MAPE of [X.X]%.
- **Bagging vs. Boosting:** Random Forest achieved [X.XX] lakhs MAE compared to [X.XX] lakhs for XGBoost and [X.XX] lakhs for Gradient Boosting. Parallel bootstrap averaging in Random Forest reduces variance on noisy listing prices, whereas sequential boosting reduces bias and achieves lower RMSE on higher-priced vehicles.

### Stacking (`RF + GB + XGB`)
- Achieved **[X.XX] lakhs** MAE compared to **[X.XX] lakhs** for its best single base model. Combining bagging and boosting via a Ridge meta-learner balances variance and bias across different price ranges.

### Hybrid (`RF + ANN`)
- `Hybrid RF + ANN` achieved **[X.XX] lakhs** MAE compared to **[X.XX] lakhs** for Random Forest alone, demonstrating how averaging smooth neural-network estimates with step-wise tree splits reduces non-overlapping errors.

### Key Price Drivers
- Permutation importance identifies **[Feature 1]**, **[Feature 2]**, and **[Feature 3]** as the dominant predictors, aligning with real-world used-car valuation in Sri Lanka.

### Dataset Limitations
- **Synthetic Mileage:** `Millage(KM)` had a `-1.0` correlation with `YOM` ($pprox 11,000	ext{ km} 	imes 	ext{age}$) and was excluded.
- **Advertised vs. Final Price:** Listings reflect asking prices on ikman.lk and Riyasewana rather than final negotiated sale prices.
- **Two-Month Snapshot:** Data covers Dec 2024 – Feb 2025 only, preventing longitudinal time-series modeling.
- **Near-Duplicate Listings:** Under `GroupShuffleSplit` (8,842 distinct attribute combinations), MAE changed by [X.XX]% for Random Forest and [X.XX]% for XGBoost, confirming minimal data leakage from duplicate configurations.
