"""Step 3b: simple baselines on the SAR-CLD-2024 split.

  1. Majority class: always predicts the biggest class (the "do nothing" score).
  2. Random forest on colour + texture features (src/features.py).

The CNN in Step 4 must beat both. Outputs in results/:
  baseline_metrics.csv          accuracy + macro-F1 on validation and test
  baseline_per_class_recall.csv recall per class on the test split
  confusion_rf_test.png         confusion matrix of the random forest (test)

Run from the project folder:  python src/baselines.py
"""
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from evaluate import compute_metrics, per_class_recall, plot_confusion_matrix
from features import features_for

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/splits"
RESULTS = ROOT / "results"
SEED = 42


def load_split(name):
    """Features (X) and labels (y) for one split."""
    df = pd.read_csv(SPLITS / f"{name}.csv")
    return features_for(df["path"].tolist()), df["class"].to_numpy()


def make_models():
    """The two baselines, with fixed seeds."""
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",  # balanced: imbalance fix
            random_state=SEED, n_jobs=-1),
    }


def main():
    print("Loading features ...")
    data = {name: load_split(name) for name in ["train", "val", "test"]}
    X_train, y_train = data["train"]
    classes = sorted(set(y_train))

    metric_rows, recall_columns = [], {}
    for model_name, model in make_models().items():
        print(f"Training {model_name} ...")
        model.fit(X_train, y_train)
        for split in ["val", "test"]:
            X, y = data[split]
            y_pred = model.predict(X)
            metric_rows.append({"model": model_name, "split": split,
                                **compute_metrics(y, y_pred, classes)})
            if split == "test":
                recall_columns[model_name] = per_class_recall(y, y_pred, classes)
                if model_name == "random_forest":
                    plot_confusion_matrix(y, y_pred, classes,
                                          "Random forest (colour + texture), test split",
                                          RESULTS / "confusion_rf_test.png")

    metrics = pd.DataFrame(metric_rows).round(3)
    metrics.to_csv(RESULTS / "baseline_metrics.csv", index=False)
    recalls = pd.DataFrame(recall_columns).round(3)
    recalls.index.name = "class"
    recalls.to_csv(RESULTS / "baseline_per_class_recall.csv")

    print("\nBaseline metrics:")
    print(metrics.to_string(index=False))
    print("\nRecall per class (test split):")
    print(recalls.to_string())


if __name__ == "__main__":
    main()
