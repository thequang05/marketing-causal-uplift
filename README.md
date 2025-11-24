# Marketing Campaign Uplift Modeling

## Objective
Build a causal inference model to identify customers who are positively influenced by digital ads, so we can optimize budget allocation and maximize incremental conversions/ROI. Instead of predicting who will convert, we focus on the difference between conversion probability with vs. without marketing exposure.

## Dataset
- 5,000 customer records, 15 features
- Key fields: channel, device, country, segment, prior visits/spend, treatment flag (`treatment_exposed`), conversion outcome
- Preprocessing: drop unused fields, one-hot encode categorical variables, 75/25 train/test split

## Methodology
- T-Learner using two Random Forest models:
  - Treatment model predicts conversion probability when a customer sees the ad
  - Control model predicts conversion probability when no ad is shown
- Uplift score = `P(conversion | treatment) - P(conversion | control)`
- Rank customers by uplift to prioritize the highest-impact targets

## Model Evaluation
- AUUC (Area Under Uplift Curve): `20,724` (significantly above random, meaning strong incremental gains)
- Qini Curve: uplift line clearly beats random strategy
- Accuracy: Treatment 89.6%, Control 89.5%
- Feature Importance: `prior_spend_180d`, `prior_visits_30d`, key channels/segments drive the uplift
- Decile & Distribution Analysis: surfaces the top 10% of customers with the strongest causal response

## Deliverables
- `uplift_results_all.csv`: full list with uplift scores and predicted probabilities
- `uplift_top_targets.csv`: customers with positive uplift (recommended targets)
- `uplift_top_10_percent.csv`: highest uplift decile
- `uplift_summary_stats.csv`: overview stats (counts, uplift distribution, accuracy, AUUC)
- Visualizations: Qini curve, feature importance plots, uplift distribution


## Business Insights
- Prioritizing top uplift segments yields far more incremental conversions than random or blanket targeting
- High-spend, frequent-visit customers show the strongest uplift
- Search/Social channels and Tech/Outdoor segments stand out as high-impact groups

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, RandomForestClassifier, Causal Inference (T-Learner)
