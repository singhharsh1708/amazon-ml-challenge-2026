
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
)

TRAIN_FILE = Path("data/training/features/train_features.parquet")
VALID_FILE = Path("data/training/features/valid_features.parquet")

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_entity_matcher.joblib"
PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.tsv"

FEATURES = [
    "cheap_score",
    "cheap_rank",
    "source_s2",
    "source_s3",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_partial",
    "name_exact",
    "address_exact",
    "name_contains",
    "address_contains",
    "name1_length",
    "name2_length",
    "address1_length",
    "address2_length",
    "name_length_diff",
    "address_length_diff",
    "name_token_overlap",
    "address_token_overlap",
]


def macro_group_f05(df, predictions):
    """
    Preliminary group-level F0.5.

    Calculates F0.5 separately for each S1 group using the
    labels available in the validation candidate dataset.
    Groups with no positive labels and no predicted matches
    receive a score of 1.

    Note: validation negatives were sampled, so this is a
    development metric, not the final challenge metric.
    """
    temp = df[["source1_entity_id", "label"]].copy()
    temp["pred"] = predictions

    scores = []

    for _, group in temp.groupby("source1_entity_id", sort=False):
        y_true = group["label"].to_numpy()
        y_pred = group["pred"].to_numpy()

        if y_true.sum() == 0 and y_pred.sum() == 0:
            scores.append(1.0)
        elif y_true.sum() == 0 or y_pred.sum() == 0:
            scores.append(0.0)
        else:
            scores.append(
                fbeta_score(
                    y_true,
                    y_pred,
                    beta=0.5,
                    zero_division=0,
                )
            )

    return float(np.mean(scores))


def main():
    print("Loading feature datasets...")

    train = pd.read_parquet(TRAIN_FILE)
    valid = pd.read_parquet(VALID_FILE)

    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(valid):,}")

    # Verify S1 group separation.
    train_ids = set(train["source1_entity_id"].unique())
    valid_ids = set(valid["source1_entity_id"].unique())

    overlap = train_ids.intersection(valid_ids)

    print(f"Train S1 groups: {len(train_ids):,}")
    print(f"Valid S1 groups: {len(valid_ids):,}")
    print(f"Overlapping S1 groups: {len(overlap):,}")

    if overlap:
        raise RuntimeError(
            "Data leakage detected: S1 groups overlap."
        )

    X_train = train[FEATURES].astype("float32")
    y_train = train["label"].astype("int8")

    X_valid = valid[FEATURES].astype("float32")
    y_valid = valid["label"].astype("int8")

    print("\nTraining class counts:")
    print(y_train.value_counts().to_string())

    print("\nValidation class counts:")
    print(y_valid.value_counts().to_string())

    # Handle the positive/negative class imbalance.
    positives = int(y_train.sum())
    negatives = len(y_train) - positives

    scale_pos_weight = negatives / max(positives, 1)

    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )

    print("\nStarting LightGBM training...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=True,
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    print("\nTraining completed.")
    print("Best iteration:", model.best_iteration_)

    print("\nPredicting validation probabilities...")

    probabilities = model.predict_proba(
        X_valid,
        num_iteration=model.best_iteration_,
    )[:, 1]

    # Basic pair-level ranking metrics.
    ap = average_precision_score(y_valid, probabilities)
    auc = roc_auc_score(y_valid, probabilities)

    print("\n--- VALIDATION RANKING METRICS ---")
    print(f"Average precision: {ap:.6f}")
    print(f"ROC AUC:           {auc:.6f}")

    # Find a preliminary threshold using validation F0.5.
    # This is for development only; do not treat it as
    # a final challenge threshold.
    thresholds = np.arange(0.05, 0.96, 0.05)

    best_threshold = 0.5
    best_f05 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        score = fbeta_score(
            y_valid,
            predictions,
            beta=0.5,
            zero_division=0,
        )

        if score > best_f05:
            best_f05 = score
            best_threshold = float(threshold)

    print("\n--- PRELIMINARY THRESHOLD ---")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Pair-level F0.5: {best_f05:.6f}")

    final_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print("\n--- PAIR-LEVEL METRICS ---")
    print(
        "Precision:",
        precision_score(
            y_valid,
            final_predictions,
            zero_division=0,
        ),
    )
    print(
        "Recall:",
        recall_score(
            y_valid,
            final_predictions,
            zero_division=0,
        ),
    )

    group_f05 = macro_group_f05(
        valid,
        final_predictions,
    )

    print("\n--- PRELIMINARY GROUP METRIC ---")
    print(f"Macro group F0.5: {group_f05:.6f}")

    # Save validation predictions for analysis.
    results = valid[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "source_label",
            "label",
            "cheap_rank",
        ]
    ].copy()

    results["match_probability"] = probabilities
    results["predicted_match"] = final_predictions

    results.to_csv(
        PREDICTIONS_PATH,
        sep="\t",
        index=False,
    )

    # Save model and selected threshold.
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": best_threshold,
            "best_iteration": model.best_iteration_,
        },
        MODEL_PATH,
    )

    print("\n--- FEATURE IMPORTANCE ---")

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    print(importance.to_string(index=False))

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print("\nTraining pipeline completed.")


if __name__ == "__main__":
    main()