"""Evaluation: shared metric helpers + Step 5 (external test, confidence rule).

The helpers at the top (compute_metrics, per_class_recall, plot_confusion_matrix)
are also used by baselines.py and train_cnn.py.

Running this file does Step 5 (after Step 4 trained the CNNs):
  1. picks the main model by VALIDATION macro-F1 (not by EWU, to keep that test honest)
  2. external test on EWU photos of the shared classes, for every trained CNN
  3. confidence thresholds chosen on the validation set (high / medium / low)
  4. coverage vs. accuracy curve, and how the rule treats EWU photos of unseen diseases
  5. saves models/cottoncare_model.pt (+ models/model_info.json) for the app

Run from the project folder:  python src/evaluate.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

TARGET_ACCURACY = 0.98   # "high" answers must be at least this accurate on validation
LOW_THRESHOLD = 0.5      # best guess under 50% = "low": ask for a clearer photo

TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
SERIES_COLORS = ["#2a78d6", "#eb6834"]  # blue, orange


# ---------- Shared helpers ----------

def compute_metrics(y_true, y_pred, classes):
    """Accuracy and macro-F1 (every class counts equally, good for imbalance)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
    }


def per_class_recall(y_true, y_pred, classes):
    """Recall per class: of all leaves of class X, how many were found."""
    values = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return pd.Series(values, index=classes)


def plot_confusion_matrix(y_true, y_pred, classes, title, out_file, row_classes=None):
    """Rows = true class, columns = predicted. Colour = share of the row, text = count.
    row_classes: show only these true classes as rows (e.g. the 2 shared EWU classes)."""
    counts = confusion_matrix(y_true, y_pred, labels=classes)
    rows = row_classes or classes
    counts = counts[[classes.index(c) for c in rows]]
    shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.0 * len(classes) + 2.5, 0.8 * len(rows) + 2))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(rows)):
        for j in range(len(classes)):
            ink = "white" if shares[i, j] > 0.55 else TEXT_COLOR
            ax.text(j, i, counts[i, j], ha="center", va="center", color=ink, fontsize=9)
    ax.set_xticks(np.arange(len(classes)), classes, rotation=40, ha="right", fontsize=9,
                  color=TEXT_COLOR)
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=9, color=TEXT_COLOR)
    ax.set_xlabel("Predicted class", color=MUTED_COLOR)
    ax.set_ylabel("True class", color=MUTED_COLOR)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_COLOR)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# ---------- Loading models and predicting ----------

def load_checkpoint(path):
    """Load a saved CNN (our own file, so weights_only=False is safe)."""
    from train_cnn import build_model
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_paths(model, ckpt, paths, batch_size=64):
    """Softmax probabilities for a list of image paths (relative to the project)."""
    from train_cnn import make_transforms
    _, eval_tf = make_transforms(augment=False)
    probs = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for p in paths[start:start + batch_size]:
            with Image.open(ROOT / p) as img:
                batch.append(eval_tf(img.convert("RGB")))
        probs.append(torch.softmax(model(torch.stack(batch)), dim=1).numpy())
    return np.concatenate(probs)


def read_probs(name, split):
    """Probabilities saved by train_cnn.py: returns (table, probs matrix, class list)."""
    table = pd.read_csv(RESULTS / f"probs_{name}_{split}.csv")
    prob_cols = [c for c in table.columns if c.startswith("prob_")]
    return table, table[prob_cols].to_numpy(), [c[len("prob_"):] for c in prob_cols]


def choose_main_model():
    """The CNN run with the best validation macro-F1 (quick smoke tests ignored)."""
    runs = pd.read_csv(RESULTS / "cnn_metrics.csv")
    val = runs[(runs["split"] == "val") & ~runs["name"].str.endswith("_quick")]
    best = val.sort_values("macro_f1", ascending=False).iloc[0]
    names = [n.removeprefix("cnn_") for n in val["name"]]
    return best["name"].removeprefix("cnn_"), names


# ---------- External test ----------

def external_test(names):
    """EWU photos of the shared classes vs. the same classes in the SAR-CLD test split."""
    ewu = pd.read_csv(SPLITS / "ewu_external.csv")
    shared = sorted(ewu[ewu["shared_with_sar"]]["class"].unique())
    rows = []
    for name in names:
        model, ckpt = load_checkpoint(MODELS / f"cnn_{name}.pt")
        classes = ckpt["classes"]
        print(f"  predicting {len(ewu)} EWU photos with {name} ...")
        probs = predict_paths(model, ckpt, ewu["path"].tolist())
        table = ewu.copy()
        table["pred"] = np.array(classes)[probs.argmax(1)]
        table["confidence"] = probs.max(1)
        for i, c in enumerate(classes):
            table[f"prob_{c}"] = probs[:, i]
        table.to_csv(RESULTS / f"probs_{name}_ewu.csv", index=False)

        sar_test, _, _ = read_probs(name, "test")
        sar_part = sar_test[sar_test["true"].isin(shared)]
        ewu_part = table[table["shared_with_sar"]]
        for dataset, y_true, y_pred in [("SAR-CLD test", sar_part["true"], sar_part["pred"]),
                                        ("EWU external", ewu_part["class"], ewu_part["pred"])]:
            recall = per_class_recall(y_true, y_pred, shared)
            rows.append({"model": f"cnn_{name}", "dataset": dataset, "images": len(y_true),
                         **compute_metrics(y_true, y_pred, shared),
                         **{f"recall_{c}": recall[c] for c in shared}})
    result = pd.DataFrame(rows).round(3)
    result.to_csv(RESULTS / "external_test.csv", index=False)
    return result, shared


# ---------- Confidence rule ----------

def choose_thresholds(confidence, correct):
    """T_high: lowest threshold whose accepted answers reach TARGET_ACCURACY (on validation).
    T_low: fixed LOW_THRESHOLD (the best guess is less likely right than wrong)."""
    t_high = 1.0
    for t in np.sort(np.unique(confidence)):
        if correct[confidence >= t].mean() >= TARGET_ACCURACY:
            t_high = float(t)
            break
    t_low = min(LOW_THRESHOLD, t_high)
    return {"high": round(t_high, 4), "low": round(t_low, 4)}


def confidence_band(confidence, thresholds):
    """high = diagnosis, medium = 'possibly A or B', low = ask for a better photo."""
    return np.where(confidence >= thresholds["high"], "high",
                    np.where(confidence >= thresholds["low"], "medium", "low"))


def band_summary(name, probs, true, classes, thresholds):
    """Share of each band, accuracy of 'high' answers, top-2 accuracy of 'medium' ones."""
    conf = probs.max(1)
    band = confidence_band(conf, thresholds)
    pred = np.array(classes)[probs.argmax(1)]
    top2 = np.array(classes)[np.argsort(-probs, axis=1)[:, :2]]
    known = np.isin(true, classes)  # unseen diseases have no correct answer
    in_top2 = (top2 == np.asarray(true)[:, None]).any(1)
    row = {"dataset": name, "images": len(conf)}
    for b in ["high", "medium", "low"]:
        row[f"share_{b}"] = (band == b).mean()
    if known.all():
        row["accuracy_all"] = (pred == true).mean()
        row["accuracy_high"] = (pred == true)[band == "high"].mean() if (band == "high").any() else np.nan
        row["top2_accuracy_medium"] = in_top2[band == "medium"].mean() if (band == "medium").any() else np.nan
    return row


def coverage_curve(confidence, correct):
    """For each threshold: share of photos answered (coverage) and their accuracy."""
    rows = []
    for t in np.linspace(0, 1, 101):
        mask = confidence >= t
        rows.append({"threshold": t, "coverage": mask.mean(),
                     "accuracy": correct[mask].mean() if mask.any() else np.nan})
    return pd.DataFrame(rows)


def plot_coverage(curves, thresholds, out_file):
    """Accuracy vs. coverage for SAR-CLD test and EWU, with the chosen T_high marked."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for color, (label, (curve, conf, correct)) in zip(SERIES_COLORS, curves.items()):
        ax.plot(curve["coverage"] * 100, curve["accuracy"] * 100, color=color, linewidth=2,
                label=label)
        mask = conf >= thresholds["high"]
        x, y = mask.mean() * 100, correct[mask].mean() * 100
        ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor="white",
                markeredgewidth=2, zorder=3)
        ax.annotate(f"{label}: answers {x:.0f}%, {y:.1f}% correct", (x, y),
                    textcoords="offset points", xytext=(-8, 9), ha="right",
                    fontsize=8, color=TEXT_COLOR)
    ax.set_xlabel("Coverage: photos that get a 'high' diagnosis (%)", color=MUTED_COLOR)
    ax.set_ylabel("Accuracy on those photos (%)", color=MUTED_COLOR)
    ax.set_title(f"Coverage vs. accuracy (dots: chosen threshold {thresholds['high']:.2f})",
                 loc="left", fontsize=11, color=TEXT_COLOR)
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 106)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def confidence_rule(main):
    """Choose thresholds on validation; evaluate them on test, EWU shared and EWU unseen."""
    val, val_probs, classes = read_probs(main, "val")
    val_correct = (val["true"] == val["pred"]).to_numpy()
    thresholds = choose_thresholds(val_probs.max(1), val_correct)

    test, test_probs, _ = read_probs(main, "test")
    ewu = pd.read_csv(RESULTS / f"probs_{main}_ewu.csv")
    ewu_probs = ewu[[f"prob_{c}" for c in classes]].to_numpy()
    shared, unseen = ewu["shared_with_sar"].to_numpy(), ~ewu["shared_with_sar"].to_numpy()

    summary = pd.DataFrame([
        band_summary("SAR-CLD validation", val_probs, val["true"].to_numpy(), classes, thresholds),
        band_summary("SAR-CLD test", test_probs, test["true"].to_numpy(), classes, thresholds),
        band_summary("EWU shared classes", ewu_probs[shared], ewu["class"].to_numpy()[shared],
                     classes, thresholds),
        band_summary("EWU unseen diseases", ewu_probs[unseen], ewu["class"].to_numpy()[unseen],
                     classes, thresholds),
    ]).round(3)
    summary.to_csv(RESULTS / "confidence_bands.csv", index=False)

    test_correct = (test["true"] == test["pred"]).to_numpy()
    ewu_correct = (ewu["class"] == ewu["pred"]).to_numpy()[shared]
    curves = {
        "SAR-CLD test": (coverage_curve(test_probs.max(1), test_correct),
                         test_probs.max(1), test_correct),
        "EWU (shared classes)": (coverage_curve(ewu_probs[shared].max(1), ewu_correct),
                                 ewu_probs[shared].max(1), ewu_correct),
    }
    pd.concat([c.assign(dataset=k) for k, (c, _, _) in curves.items()]).round(4).to_csv(
        RESULTS / "coverage_curve.csv", index=False)
    plot_coverage(curves, thresholds, RESULTS / "coverage_curve.png")

    unseen_preds = pd.crosstab(ewu["class"][unseen], ewu["pred"][unseen])
    unseen_preds.to_csv(RESULTS / "ewu_unseen_predictions.csv")
    return thresholds, summary, unseen_preds


# ---------- Final model for the app ----------

def save_final_model(main, thresholds, external, reason):
    """Copy the main checkpoint, add the thresholds, and write a readable info file."""
    _, ckpt = load_checkpoint(MODELS / f"cnn_{main}.pt")
    ckpt["threshold"] = thresholds
    ckpt["source_run"] = main
    torch.save(ckpt, MODELS / "cottoncare_model.pt")
    info = {"model_file": "cottoncare_model.pt", "source_run": main,
            "selection_reason": reason, "arch": ckpt["arch"],
            "classes": ckpt["classes"], "img_size": ckpt["img_size"],
            "thresholds": thresholds, "target_accuracy_high": TARGET_ACCURACY,
            "low_threshold_rule": f"top-1 confidence < {LOW_THRESHOLD}", "seed": ckpt["seed"],
            "test_metrics": {k: round(float(v), 4) for k, v in ckpt["metrics"]["test"].items()},
            "external_test": external[external["model"] == f"cnn_{main}"].to_dict("records")}
    (MODELS / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 5: external test + confidence rule")
    parser.add_argument("--main", default=None,
                        help="run to use in the app (default: best validation macro-F1)")
    parser.add_argument("--reason", default="best validation macro-F1",
                        help="why this run was chosen (saved in model_info.json)")
    args = parser.parse_args()

    best_val_run, all_runs = choose_main_model()
    main_run = args.main or best_val_run
    print(f"Best validation run: {best_val_run}   all runs: {all_runs}")
    print(f"Main model for the app: {main_run} ({args.reason})")

    print("\n1) External test (EWU) ...")
    external, shared = external_test(all_runs)
    print(external.to_string(index=False))

    _, ckpt = load_checkpoint(MODELS / f"cnn_{main_run}.pt")
    ewu = pd.read_csv(RESULTS / f"probs_{main_run}_ewu.csv")
    ewu_shared = ewu[ewu["shared_with_sar"]]
    plot_confusion_matrix(ewu_shared["class"], ewu_shared["pred"], ckpt["classes"],
                          f"MobileNetV3-{ckpt['arch']} on EWU photos (external test)",
                          RESULTS / f"confusion_cnn_{main_run}_ewu.png", row_classes=shared)

    print("\n2) Confidence rule (thresholds from the validation set) ...")
    thresholds, bands, unseen = confidence_rule(main_run)
    print(f"Thresholds: high >= {thresholds['high']}, low < {thresholds['low']}")
    print(bands.to_string(index=False))
    print("\nWhat the model says for EWU diseases it never saw (counts):")
    print(unseen.to_string())

    save_final_model(main_run, thresholds, external, args.reason)
    print("\nSaved models/cottoncare_model.pt and models/model_info.json")


if __name__ == "__main__":
    main()
