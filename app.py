"""
Obesity Level Classifier - Streamlit demo app.

Loads five pre-trained scikit-learn pipelines (trained in notebook/obesity_classification.ipynb
on the UCI "Estimation of Obesity Levels" dataset) and scores a user-uploaded test CSV.
No retraining happens here - the app only loads saved models and evaluates them.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

st.set_page_config(page_title="Obesity Level Classifier", page_icon="⚖️", layout="wide")

MODEL_DIR = Path(__file__).parent / "model"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

MODEL_COLORS = {
    "Logistic Regression": "#4C72B0",
    "Decision Tree": "#DD8452",
    "K-Nearest Neighbors": "#55A868",
    "Naive Bayes": "#C44E52",
    "Support Vector Machine": "#8172B2",
}

CLASS_NOTES = {
    "Insufficient_Weight": "BMI below the healthy range",
    "Normal_Weight": "BMI within the healthy range",
    "Overweight_Level_I": "Mildly above the healthy range",
    "Overweight_Level_II": "Moderately above the healthy range",
    "Obesity_Type_I": "Obesity, class I",
    "Obesity_Type_II": "Obesity, class II",
    "Obesity_Type_III": "Obesity, class III (severe)",
}


@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    pipelines = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    return pipelines, label_encoder, feature_cols


def score_model(pipeline, X, y_true_encoded):
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    auc = roc_auc_score(y_true_encoded, y_proba, multi_class="ovr", average="macro")
    metrics = {
        "Accuracy": accuracy_score(y_true_encoded, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true_encoded, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true_encoded, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true_encoded, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true_encoded, y_pred),
    }
    return metrics, y_pred


def radar_chart(comparison_df):
    n = len(METRIC_NAMES)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for model_name, row in comparison_df.iterrows():
        values = row[METRIC_NAMES].tolist()
        values += values[:1]
        color = MODEL_COLORS.get(model_name)
        ax.plot(angles, values, linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_NAMES)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def class_distribution_chart(df, target_col):
    counts = df[target_col].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def grouped_bar_chart(comparison_df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(METRIC_NAMES))
    n_models = len(comparison_df)
    width = 0.8 / n_models
    for i, (model_name, row) in enumerate(comparison_df.iterrows()):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, row[METRIC_NAMES].values, width,
               label=model_name, color=MODEL_COLORS.get(model_name))
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


pipelines, label_encoder, feature_cols = load_artifacts()
target_col = "NObeyesdad"
n_classes = len(label_encoder.classes_)

st.title("⚖️ Obesity Level Classifier")
st.caption(
    "Demo app for BITS WILP M.Tech (AIML) Machine Learning - Assignment 2. "
    "Five classifiers were trained offline on the UCI *Estimation of Obesity Levels Based on "
    "Eating Habits and Physical Condition* dataset (2,111 rows, 16 lifestyle/physical features, "
    "7-class obesity category target). This app only loads the saved models - it never retrains."
)

with st.expander("ℹ️ Problem statement & dataset", expanded=False):
    st.markdown(
        "**Problem statement** — Predict a person's obesity category (one of 7 classes, "
        "from `Insufficient_Weight` to `Obesity_Type_III`) from their eating habits, physical "
        "activity, and basic physical measurements. This is a multi-class classification "
        "problem: five different classifiers are trained on the same dataset and compared on "
        "six evaluation metrics, then wrapped in this app, which scores an uploaded test CSV "
        "using the already-trained models (no retraining happens here)."
    )
    st.markdown(
        "**Dataset** — UCI *Estimation of Obesity Levels Based on Eating Habits and Physical "
        "Condition* (id 544): 2,111 instances, 16 features — a mix of continuous measurements "
        "(`Age`, `Height`, `Weight`), lifestyle scores (`FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`), "
        "and categorical/binary habits (`Gender`, `family_history_with_overweight`, `FAVC`, "
        "`CAEC`, `SMOKE`, `SCC`, `CALC`, `MTRANS`). Target `NObeyesdad` has 7 fairly balanced "
        "classes (roughly 13-16% of rows each)."
    )

with st.expander("What does the target column mean?"):
    legend_df = pd.DataFrame(
        {"Class": list(CLASS_NOTES.keys()), "Meaning": list(CLASS_NOTES.values())}
    )
    st.table(legend_df)

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader(
        "Upload a test CSV",
        type=["csv"],
        help="Must contain all 16 feature columns plus the `NObeyesdad` target column.",
    )
    sample_path = Path(__file__).parent / "test_data.csv"
    if sample_path.exists():
        st.download_button(
            "⬇️ Download sample test_data.csv",
            data=sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.divider()
    model_choice = st.selectbox(
        "Model for detailed view", list(MODEL_FILES.keys()),
        disabled=uploaded_file is None,
    )
    normalize_cm = st.checkbox("Normalize confusion matrix (row %)", value=False)

if uploaded_file is None:
    st.info("⬆️ Upload a CSV from the sidebar to see model predictions and evaluation metrics.")
    st.stop()

df = pd.read_csv(uploaded_file)

missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded file is missing required columns: {missing_cols}")
    st.stop()

X_uploaded = df[feature_cols]
unseen_labels = set(df[target_col].unique()) - set(label_encoder.classes_)
if unseen_labels:
    st.error(f"Uploaded target contains labels the models were not trained on: {unseen_labels}")
    st.stop()
y_uploaded_encoded = label_encoder.transform(df[target_col])

st.success(f"✅ Loaded {len(df)} rows from **{uploaded_file.name}**.")

comparison_rows = []
predictions_by_model = {}
for name, pipeline in pipelines.items():
    m, y_pred = score_model(pipeline, X_uploaded, y_uploaded_encoded)
    m["Model"] = name
    comparison_rows.append(m)
    predictions_by_model[name] = y_pred

comparison_df = pd.DataFrame(comparison_rows).set_index("Model")[METRIC_NAMES]
best_per_metric = comparison_df.max(axis=0)

tab_overview, tab_model, tab_compare = st.tabs(
    ["📊 Dataset Overview", "🎯 Selected Model", "📈 Compare All Models"]
)

with tab_overview:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Class distribution in the uploaded data**")
        st.pyplot(class_distribution_chart(df, target_col))
    with right:
        st.markdown("**Preview**")
        st.dataframe(df.head(10), height=320)

with tab_model:
    st.subheader(f"Results — {model_choice}")
    chosen_metrics = comparison_df.loc[model_choice]
    y_pred = predictions_by_model[model_choice]

    metric_cols = st.columns(6)
    for col, metric_name in zip(metric_cols, METRIC_NAMES):
        value = chosen_metrics[metric_name]
        gap = value - best_per_metric[metric_name]
        delta = "best of 5" if gap >= -1e-9 else f"{gap:+.3f} vs best"
        col.metric(metric_name, f"{value:.3f}", delta=delta)

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion Matrix**" + (" (row %)" if normalize_cm else ""))
        cm = confusion_matrix(y_uploaded_encoded, y_pred, labels=range(n_classes))
        if normalize_cm:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            fmt = ".1f"
        else:
            cm_display = cm
            fmt = "d"
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_display, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_uploaded_encoded, y_pred, target_names=label_encoder.classes_,
            zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), height=400)

    predictions_df = df.copy()
    predictions_df[f"predicted_{target_col}"] = label_encoder.inverse_transform(y_pred)
    st.download_button(
        f"⬇️ Download predictions ({model_choice})",
        data=predictions_df.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

with tab_compare:
    st.subheader("All five models on this uploaded file")
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(grouped_bar_chart(comparison_df))
    with right:
        st.pyplot(radar_chart(comparison_df))
    st.dataframe(
        comparison_df.round(4).style.highlight_max(axis=0, color="#c6f6c6"),
        use_container_width=True,
    )
