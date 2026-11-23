
### Overall Performance
- **Best Model:** Stacking (RF+GB+XGB) (Stacking) achieved the lowest error with an MAE of 5.66 lakhs and MAPE of 13.2%.
- **Bagging vs. Boosting:** Random Forest achieved 5.83 lakhs MAE compared to 5.92 lakhs for XGBoost and 7.30 lakhs for Gradient Boosting. Parallel bootstrap averaging in Random Forest reduces variance on noisy listing prices, whereas sequential boosting reduces bias and achieves lower RMSE on higher-priced vehicles.

### Stacking (`RF + GB + XGB`)
- Achieved **5.66 lakhs** MAE compared to **5.83 lakhs** for its best single base model. Combining bagging and boosting via a Ridge meta-learner balances variance and bias across different price ranges.

### Hybrid (`RF + ANN`)
- `Hybrid RF + ANN` achieved **5.72 lakhs** MAE compared to **5.83 lakhs** for Random Forest alone, demonstrating how averaging smooth neural-network estimates with step-wise tree splits reduces non-overlapping errors.

### Key Price Drivers
- Permutation importance identifies **Age**, **Brand**, and **Gear** as the dominant predictors, aligning with real-world used-car valuation in Sri Lanka.

### Dataset Limitations
- **Synthetic Mileage:** `Millage(KM)` had a `-1.0` correlation with `YOM` ($\approx 11,000\text{ km} \times \text{age}$) and was excluded.
- **Advertised vs. Final Price:** Listings reflect asking prices on ikman.lk and Riyasewana rather than final negotiated sale prices.
- **Two-Month Snapshot:** Data covers Dec 2024 – Feb 2025 only, preventing longitudinal time-series modeling.
- **Near-Duplicate Listings:** Under `GroupShuffleSplit` (8,842 distinct attribute combinations), MAE changed minimally across grouped splits, confirming low data leakage from duplicate configurations.
