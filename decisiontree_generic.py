#!/usr/bin/env python3
"""
decisiontree_generic.py

CLI untuk membangun pipeline DecisionTreeClassifier end-to-end:
- Load CSV (UCI winequality-like)
- Buat label binary (edible vs poisonous) dari kolom 'quality'
  (default: quality >= 6 -> edible; else poisonous)
- Stratified train/test split (random_state=42)
- Preprocessing dengan ColumnTransformer:
    - categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
    - numeric: SimpleImputer(median)
- Pipeline + DecisionTreeClassifier(random_state=42)
- Cost-complexity pruning tuning (GridSearchCV, cv=5) pada clf__ccp_alpha
- Jika hasil terbaik menghasilkan > node_limit nodes, script mencoba memilih alpha
  yang menghasilkan tree <= node_limit bila memungkinkan
- Evaluasi: accuracy, classification report, confusion matrix, feature importances, precision–recall curve
- Visualisasi pohon terpruned
- Simpan artefak ke reports/ dan models/ (joblib)
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")


def make_dirs(output_dir: Path, models_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def make_label(df: pd.DataFrame, threshold: int = 6, quality_col: str = "quality") -> pd.Series:
    # Default rule: quality >= threshold => edible (1); else poisonous (0)
    lab = (df[quality_col].astype(float) >= threshold).map({True: 1, False: 0})
    return lab


def build_preprocessor(df: pd.DataFrame):
    # detect categorical cols (object or category)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # ensure we don't include the target
    if "quality" in categorical_cols:
        categorical_cols.remove("quality")
    numeric_cols = [c for c in df.columns if c not in categorical_cols and c != "quality"]

    # numeric transformer: impute median
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # categorical transformer: impute most frequent + onehot
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor, categorical_cols, numeric_cols


def get_feature_names(preprocessor, categorical_cols, numeric_cols):
    # After fitting preprocessor we can get proper feature names
    names = []
    if "cat" in preprocessor.named_transformers_:
        cat_pipe = preprocessor.named_transformers_["cat"]
        if hasattr(cat_pipe.named_steps["onehot"], "get_feature_names_out"):
            cat_names = cat_pipe.named_steps["onehot"].get_feature_names_out(categorical_cols).tolist()
        else:
            cat_names = []
        names.extend(cat_names)
    names.extend(numeric_cols)
    return names


def fit_initial_tree_and_get_alphas(preprocessor, X_train, y_train, random_state=42):
    # Fit preprocessor and transform X_train
    preprocessor.fit(X_train)
    X_tr = preprocessor.transform(X_train)
    # Fit an unpruned tree to compute ccp_alphas
    base_tree = DecisionTreeClassifier(random_state=random_state)
    base_tree.fit(X_tr, y_train)
    path = base_tree.cost_complexity_pruning_path(X_tr, y_train)
    ccp_alphas = np.unique(path.ccp_alphas)
    # Exclude the maximum alpha that prunes the tree to a single node
    if ccp_alphas.size > 1:
        ccp_alphas = ccp_alphas[:-1]
    # choose a grid of alphas (include small and derived alphas)
    ccp_grid = np.unique(np.concatenate([
        ccp_alphas,
        np.linspace(0.0, ccp_alphas.max() if ccp_alphas.size else 0.01, num=10)
    ]))
    ccp_grid = np.sort(ccp_grid)
    return ccp_grid, preprocessor, X_tr


def find_pruned_model_with_node_limit(best_pipeline, X_train, y_train, preprocessor, ccp_candidates, node_limit=25, random_state=42):
    """
    If the best model has > node_limit nodes, try larger ccp_alpha values
    to get a pruned tree with <= node_limit. Try alphas from large to small.
    """
    clf = best_pipeline.named_steps["clf"]
    try:
        current_nodes = clf.tree_.node_count
    except Exception:
        current_nodes = None

    if current_nodes is not None and current_nodes <= node_limit:
        return best_pipeline, clf

    ccp_sorted = np.sort(ccp_candidates)
    for alpha in ccp_sorted[::-1]:
        tmp_clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=float(alpha))
        tmp_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", tmp_clf)
        ])
        tmp_pipeline.fit(X_train, y_train)
        nodes = tmp_pipeline.named_steps["clf"].tree_.node_count
        if nodes <= node_limit:
            return tmp_pipeline, tmp_pipeline.named_steps["clf"]
    return best_pipeline, best_pipeline.named_steps["clf"]


def evaluate_and_report(model_pipeline, X_test, y_test, feature_names, output_dir: Path, prefix="best"):
    Xt = model_pipeline.named_steps["preprocessor"].transform(X_test)
    clf = model_pipeline.named_steps["clf"]
    y_pred = clf.predict(Xt)
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    # feature importances
    importances = {}
    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        if len(fi) == len(feature_names):
            importances = dict(zip(feature_names, fi.round(6).tolist()))
        else:
            importances = {"note": "feature_importances length mismatch"}
    else:
        importances = {"note": "no feature_importances_ available"}

    metrics = {
        "accuracy": float(acc),
        "classification_report": cr,
        "confusion_matrix": cm,
        "n_nodes": int(clf.tree_.node_count) if hasattr(clf, "tree_") else None,
        "feature_importances": importances
    }

    # write metrics json
    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics, y_pred


def plot_precision_recall(model_pipeline, X_test, y_test, output_dir: Path, prefix="best"):
    pre = model_pipeline.named_steps["preprocessor"]
    clf = model_pipeline.named_steps["clf"]
    Xt = pre.transform(X_test)
    if len(np.unique(y_test)) == 2:
        y_score = clf.predict_proba(Xt)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xt)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"AP={ap:0.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend()
        plt.grid(True)
        out = output_dir / f"{prefix}_precision_recall.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        return str(out)
    return None


def plot_confusion_matrix(cm, output_dir: Path, prefix="best"):
    cm_arr = np.array(cm)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    classes = ["poisonous", "edible"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(j, i, format(cm_arr[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm_arr[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out = output_dir / f"{prefix}_confusion_matrix.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return str(out)


def plot_tree_diagram(model_pipeline, feature_names, output_dir: Path, prefix="best", max_depth=4):
    clf = model_pipeline.named_steps["clf"]
    plt.figure(figsize=(20, 12))
    plot_tree(clf, feature_names=feature_names, filled=True, impurity=True, rounded=True, fontsize=8)
    out = output_dir / f"{prefix}_tree.png"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    return str(out)


def save_models(pipeline, preprocessor, classifier, models_dir: Path):
    joblib.dump(pipeline, models_dir / "tree_pipeline.joblib")
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(classifier, models_dir / "classifier.joblib")


def main(args):
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    make_dirs(output_dir, models_dir)

    df = load_data(data_path)
    # create label
    y = make_label(df, threshold=args.quality_threshold)
    # drop rows with missing quality
    mask_valid = ~df["quality"].isna()
    df = df.loc[mask_valid].reset_index(drop=True)
    y = y.loc[mask_valid].reset_index(drop=True)

    # drop non-feature columns if present
    X = df.drop(columns=["quality"])

    # train/test split stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # build preprocessor
    preprocessor, categorical_cols, numeric_cols = build_preprocessor(X_train)

    # get candidate ccp_alphas by fitting initial tree on transformed train
    ccp_candidates, preprocessor_fitted, X_train_trans = fit_initial_tree_and_get_alphas(preprocessor, X_train, y_train, random_state=42)

    # pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        "clf__ccp_alpha": np.unique(np.concatenate(([0.0], ccp_candidates))).tolist()
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=args.n_jobs, scoring="f1", verbose=1)
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    best_alpha = grid.best_params_.get("clf__ccp_alpha", None)
    best_score = grid.best_score_

    # get feature names after fitting preprocessor
    feature_names = get_feature_names(best_pipe.named_steps["preprocessor"], categorical_cols, numeric_cols)

    # if too many nodes, attempt to find alpha with <= node limit
    final_pipe, final_clf = find_pruned_model_with_node_limit(
        best_pipe, X_train, y_train, best_pipe.named_steps["preprocessor"], ccp_candidates, node_limit=args.node_limit, random_state=42
    )

    # Train baseline (no pruning) for comparison
    baseline_pipe = Pipeline([
        ("preprocessor", best_pipe.named_steps["preprocessor"]),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])
    baseline_pipe.fit(X_train, y_train)

    # Evaluate baseline
    baseline_metrics, _ = evaluate_and_report(baseline_pipe, X_test, y_test, feature_names, output_dir, prefix="baseline")
    # Evaluate final
    final_metrics, y_pred = evaluate_and_report(final_pipe, X_test, y_test, feature_names, output_dir, prefix="best")

    # plots
    pr_path = plot_precision_recall(final_pipe, X_test, y_test, output_dir, prefix="best")
    cm_path = plot_confusion_matrix(final_metrics["confusion_matrix"], output_dir, prefix="best")
    tree_path = plot_tree_diagram(final_pipe, feature_names, output_dir, prefix="best")

    # save models
    save_models(final_pipe, final_pipe.named_steps["preprocessor"], final_pipe.named_steps["clf"], models_dir)

    # write summary file
    summary = {
        "best_grid_score": best_score,
        "best_alpha_from_gridsearch": float(best_alpha) if best_alpha is not None else None,
        "final_model_nodes": final_metrics["n_nodes"],
        "baseline_nodes": baseline_metrics["n_nodes"],
        "reports": {
            "baseline_metrics": str(output_dir / "baseline_metrics.json"),
            "best_metrics": str(output_dir / "best_metrics.json"),
            "precision_recall_plot": pr_path,
            "confusion_matrix_plot": cm_path,
            "tree_plot": tree_path
        }
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Done. Artifacts saved to:", output_dir, "and models to:", models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune a DecisionTreeClassifier for wine edible vs poisonous.")
    parser.add_argument("--data", type=str, required=True, help="Path to winequality CSV")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to write reports/plots/metrics")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save models (joblib)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--quality-threshold", type=int, default=6, help="Quality threshold to consider edible")
    parser.add_argument("--node-limit", type=int, default=25, help="Desired maximum number of nodes for pruned tree")
    parser.add_argument("--n-jobs", type=int, default=-1, help="n_jobs for GridSearchCV")
    args = parser.parse_args()
    main(args)
