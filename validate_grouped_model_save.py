"""Run student-grouped validation for the existing OULAD Streamlit project.

Place alongside dashboard.py and the six CSV files, then run:
    python3 validate_grouped_model.py
This script does not modify dashboard.py or the deployed app.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from dashboard import _load_tables, _build_frames

SEED = 42
ROOT = Path(__file__).resolve().parent


def make_pipeline(num, cat):
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    return ImbPipeline([
        ("preprocess", preprocess),
        ("smote", SMOTE(random_state=SEED)),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1,
        )),
    ])


def main():
    tables = _load_tables(str(ROOT))
    _, data, num, cat = _build_frames(tables)
    X = data[num + cat]
    y = data["at_risk"].astype(int)
    groups = data["id_student"]

    # Reserve a genuinely unseen set of students for final evaluation.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]
    assert set(g_train).isdisjoint(set(g_test)), "Student overlap detected!"

    print(f"Registrations: train={len(train_idx):,}; test={len(test_idx):,}")
    print(f"Unique students: train={g_train.nunique():,}; test={g_test.nunique():,}")
    print("Student overlap between train and test: 0")
    print(f"At-risk proportion: train={y_train.mean():.1%}; test={y_test.mean():.1%}")

    # Five folds within TRAINING students only. Fit preprocessing and SMOTE
    # inside each fold, so validation folds are not resampled.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_train, y_train, groups=g_train))
    for tr, va in folds:
        assert set(g_train.iloc[tr]).isdisjoint(set(g_train.iloc[va]))
    scores = cross_validate(
        make_pipeline(num, cat), X_train, y_train, groups=g_train,
        cv=folds,
        scoring={
            "accuracy": "accuracy", "precision": "precision",
            "recall": "recall", "f1": "f1", "roc_auc": "roc_auc",
        },
        n_jobs=1, error_score="raise",
    )
    fold_df = pd.DataFrame({
        "Fold": range(1, 6),
        **{metric: scores[f"test_{metric}"] for metric in
           ("accuracy", "precision", "recall", "f1", "roc_auc")},
    })
    fold_df.to_csv(ROOT / "grouped_cv_folds.csv", index=False)
    print("\n5-fold student-grouped cross-validation (training only):")
    print(fold_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCross-validation mean ± standard deviation:")
    for col in fold_df.columns[1:]:
        print(f"  {col}: {fold_df[col].mean():.4f} ± {fold_df[col].std():.4f}")

    # Refit on all training students, evaluate ONCE on held-out test students.
    model = make_pipeline(num, cat)
    model.fit(X_train, y_train)
    joblib.dump(model, ROOT / "grouped_model.joblib", compress=3)
    print("Saved grouped_model.joblib (trained on grouped training students only)")
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    final_metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }
    pd.DataFrame([final_metrics]).to_csv(ROOT / "grouped_test_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    pd.DataFrame(cm,
        index=["Actual not at risk", "Actual at risk"],
        columns=["Predicted not at risk", "Predicted at risk"],
    ).to_csv(ROOT / "grouped_test_confusion_matrix.csv")
    print("\nHELD-OUT STUDENT-GROUPED TEST RESULTS:")
    for name, value in final_metrics.items():
        print(f"  {name}: {value:.1%}")
    print("Confusion matrix [not at-risk, at-risk]:")
    print(cm)
    print("\nSaved grouped_cv_folds.csv, grouped_test_metrics.csv, grouped_test_confusion_matrix.csv")
    print("Your existing Streamlit dashboard code is unchanged.")


if __name__ == "__main__":
    main()
