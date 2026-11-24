# Customer Segmentation Project

Segments customers by **demographics** (age, income) and **purchase behavior** (RFM: Recency, Frequency, Monetary) using **K-Means clustering** in Python (scikit-learn).

## Results (sample data, 1,200 customers)

The silhouette score was highest at **k = 4**, giving four segments:

| Segment | % Customers | % Revenue | Profile | Suggested action |
|---|---|---|---|---|
| Champions | 20% | 74% | Avg age 38, high income, bought ~15 days ago, ~28 orders | VIP perks, early access, referral rewards |
| Loyal Regulars | 30% | 17% | Younger (avg 29), mid income, frequent small orders, like Sports/Fashion | Bundles and upsells to raise order value |
| At Risk | 25% | 7.5% | Older (avg 52), good income, last order ~120 days ago | Win-back emails, personalised Home & Kitchen offers |
| Lost / Low Value | 25% | 2% | Low income, ~217 days inactive, ~2 orders | Low-cost reactivation only; do not over-invest |

**Key insight:** 20% of customers generate about three quarters of revenue, so retaining Champions matters most, and At Risk customers are the best win-back opportunity.

## Charts
See the `charts/` folder: choosing k, PCA segment map, size vs revenue, profile heatmap, RFM box plots and category preferences.

## How to run
```
pip install -r requirements.txt
python segmentation.py
```
To use your own data, add a `customers.csv` with columns `Age, AnnualIncome, Recency, Frequency, Monetary` (optional: `CustomerID, Gender, PreferredCategory`).

## Method
1. Clean data, log-transform skewed Recency and Monetary, standardise features
2. Pick k with the elbow method and silhouette score
3. Fit K-Means, name segments by RFM rank, profile each segment
4. Visualise and write recommendations

## Files
- `segmentation.py` main script
- `customers.csv` input data (synthetic sample)
- `customers_segmented.csv` customers with assigned segment
- `segment_summary.csv` per-segment statistics
