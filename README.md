# Obesity Level Classification — ML Assignment 2

**Author:** Atharva Sadanand Litake (BITS ID 2026AJ05822)
**Course:** BITS WILP M.Tech (AIML) — Machine Learning, Assignment 2

## a. Problem statement

Predict a person's obesity category (one of 7 classes, from `Insufficient_Weight` to
`Obesity_Type_III`) from their eating habits, physical activity, and basic physical
measurements. This is a multi-class classification problem: five different classifiers are
trained on the same dataset and compared on six evaluation metrics, then wrapped in a
Streamlit app that scores an uploaded test CSV using the already-trained models.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) (id 544)
- **Instances:** 2,111
- **Features:** 16 — a mix of continuous (`Age`, `Height`, `Weight`), lifestyle scores
  (`FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`), and categorical/binary attributes (`Gender`,
  `family_history_with_overweight`, `FAVC`, `CAEC`, `SMOKE`, `SCC`, `CALC`, `MTRANS`).
- **Target:** `NObeyesdad` — 7 classes (`Insufficient_Weight`, `Normal_Weight`,
  `Overweight_Level_I`, `Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`,
  `Obesity_Type_III`), fairly balanced (roughly 13–16% of rows per class).
- **Split:** stratified 80/20 train/test. The raw (unencoded) 20% test split is saved as
  [`test_data.csv`](test_data.csv) — this is the file used both for offline evaluation and for
  the Streamlit app demo.

## c. GitHub Repository Link

[https://github.com/AtharvaLitake/2026AJ05822-obesity-classification](https://github.com/AtharvaLitake/2026AJ05822-obesity-classification)

## d. Models used

Each model is a scikit-learn `Pipeline` (shared `ColumnTransformer` preprocessing +
classifier), tuned with 5-fold stratified cross-validation where applicable, and evaluated on
the held-out 20% test split. Full code: [`notebook/obesity_classification.ipynb`](notebook/obesity_classification.ipynb).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9598 | 0.9974 | 0.9588 | 0.9593 | 0.9590 | 0.9531 |
| Decision Tree | 0.9149 | 0.9488 | 0.9152 | 0.9117 | 0.9130 | 0.9007 |
| KNN | 0.8582 | 0.9577 | 0.8523 | 0.8528 | 0.8459 | 0.8362 |
| Naive Bayes | 0.5083 | 0.8372 | 0.5207 | 0.5002 | 0.4511 | 0.4471 |
| Support Vector Machine | 0.9504 | 0.9986 | 0.9500 | 0.9493 | 0.9495 | 0.9421 |

_(AUC/Precision/Recall/F1 are macro-averaged since this is a 7-class problem; MCC is the
multi-class generalized Matthews Correlation Coefficient. Numbers from `model/comparison_results.csv`,
produced by running `notebook/obesity_classification.ipynb` end-to-end with `random_state=42`.)_

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-calibrated baseline once `Weight`/`Height` are standardized; struggles most at the boundary between adjacent overweight/obesity sub-levels. |
| Decision Tree | Easily learns the `Weight`-driven thresholds between classes, but a few deep splits overfit minority adjacent classes until `max_depth`/`min_samples_leaf` are tuned down. |
| KNN | Competitive once features are scaled, since obesity classes form fairly compact clusters in `Weight`/`Height` space; sensitive to the choice of `k` and the slowest to score at inference time. |
| Naive Bayes | Clearly the weakest model (0.51 accuracy vs. 0.86-0.96 for the others) for two compounding reasons: its conditional-independence assumption is violated by `Weight` and `Height` jointly determining BMI, and — more importantly — a single `GaussianNB` is fit across *all* preprocessed features, including the one-hot encoded categorical columns (`Gender`, `CAEC`, `MTRANS`, etc.), which are binary 0/1 values. Modeling a binary indicator with a continuous Gaussian distribution is a direct mismatch with Naive Bayes' own assumptions, and this likely explains most of the performance gap rather than feature correlation alone. |
| Support Vector Machine | Matches or beats Logistic Regression after tuning `C`/`gamma`; the RBF kernel captures the mild non-linearity at class boundaries, at the cost of being the slowest model to train. |
| **Overall, Winner for your dataset?** | **Logistic Regression** — highest F1 (0.9590) and accuracy (0.9598) of the five, narrowly ahead of the Support Vector Machine (F1 0.9495, but the best AUC at 0.9986). |

## Repository structure

```
obesity-classification/
├── app.py                     # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv              # held-out raw test split (features + target)
├── data/
│   └── obesity_raw.csv        # full raw dataset
├── model/                     # saved pipelines (preprocessing + classifier) and metadata
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── label_encoder.joblib
│   ├── feature_columns.joblib
│   └── comparison_results.csv
└── notebook/
    └── obesity_classification.ipynb   # EDA, training, evaluation
```

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To regenerate `data/obesity_raw.csv`, `test_data.csv`, and the files in `model/`, open
`notebook/obesity_classification.ipynb` in Jupyter or VS Code and run all cells top to bottom
(requires `pip install jupyter` in addition to `requirements.txt`).

## Live Streamlit App Link

[https://2026aj05822-obesity-classification.streamlit.app/](https://2026aj05822-obesity-classification.streamlit.app/)
