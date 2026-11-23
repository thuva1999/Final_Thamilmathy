"""Shared evaluation helpers: metrics tables and confusion-matrix plots.

Used by baselines.py now, and by the CNN evaluation in Steps 4 and 5.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"


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


def plot_confusion_matrix(y_true, y_pred, classes, title, out_file):
    """Rows = true class, columns = predicted. Colour = share of the row, text = count."""
    counts = confusion_matrix(y_true, y_pred, labels=classes)
    shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.0 * len(classes) + 2.5, 0.8 * len(classes) + 2))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ink = "white" if shares[i, j] > 0.55 else TEXT_COLOR
            ax.text(j, i, counts[i, j], ha="center", va="center", color=ink, fontsize=9)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks, classes, rotation=40, ha="right", fontsize=9, color=TEXT_COLOR)
    ax.set_yticks(ticks, classes, fontsize=9, color=TEXT_COLOR)
    ax.set_xlabel("Predicted class", color=MUTED_COLOR)
    ax.set_ylabel("True class", color=MUTED_COLOR)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_COLOR)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
