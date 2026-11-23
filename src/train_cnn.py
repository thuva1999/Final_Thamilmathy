"""Step 4: fine-tune MobileNetV3 on the SAR-CLD-2024 split (GPU on Colab, or CPU).

Two phases (freeze early layers first, as in the midterm):
  1. "head"     : only the new classifier layer learns   (--epochs-head, lr 1e-3)
  2. "finetune" : the second half of the network also learns (--epochs-finetune, lr 1e-4)
The epoch with the best validation macro-F1 is kept.

Examples (from the project folder):
  python src/train_cnn.py --augment            # with data augmentation
  python src/train_cnn.py --no-augment         # without (for the comparison)
  python src/train_cnn.py --quick              # tiny smoke test, a few images

Outputs (NAME = e.g. small_aug):
  models/cnn_NAME.pt                   weights + class list + settings (threshold added in Step 5)
  results/cnn_metrics.csv              accuracy + macro-F1 (val, test) for every run
  results/cnn_per_class_recall_NAME.csv, results/confusion_cnn_NAME_test.png
  results/train_history_NAME.csv, results/train_curve_NAME.png
  results/probs_NAME_val.csv, results/probs_NAME_test.csv   (softmax outputs, used in Step 5)
  results/model_comparison.csv         baselines + all CNN runs on the test split
"""
import argparse
import copy
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from evaluate import compute_metrics, per_class_recall, plot_confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics


# ---------- Settings ----------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MobileNetV3 on cotton leaves")
    p.add_argument("--arch", choices=["small", "large"], default="small")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs-head", type=int, default=5)
    p.add_argument("--epochs-finetune", type=int, default=10)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="smoke test on 8 images per class")
    p.add_argument("--name", default=None, help="run name (default: ARCH_aug / ARCH_noaug)")
    args = p.parse_args()
    if args.name is None:
        args.name = f"{args.arch}_{'aug' if args.augment else 'noaug'}"
        if args.quick:
            args.name += "_quick"
    return args


def set_seed(seed):
    """Same seed -> same results (as far as the hardware allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Data ----------

def make_transforms(augment):
    """Training transform (with or without augmentation) and evaluation transform."""
    finish = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), *finish])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),  # zoom in a little
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        *finish,
    ])
    return train_tf, eval_tf


class LeafDataset(Dataset):
    """Reads the images listed in a split CSV (path, class)."""

    def __init__(self, df, classes, transform):
        self.paths = df["path"].tolist()
        self.labels = [classes.index(c) for c in df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(ROOT / self.paths[i]) as img:
            x = self.transform(img.convert("RGB"))
        return x, self.labels[i]


def load_splits(quick):
    """Train/val/test tables; --quick keeps only 8 images per class."""
    splits = {name: pd.read_csv(SPLITS / f"{name}.csv") for name in ["train", "val", "test"]}
    if quick:
        splits = {k: v.groupby("class").head(8).reset_index(drop=True) for k, v in splits.items()}
    return splits


def make_loaders(splits, classes, augment, batch_size, device, seed):
    train_tf, eval_tf = make_transforms(augment)
    workers = 2 if device.type == "cuda" else 0  # 0 is safest on Windows
    gen = torch.Generator().manual_seed(seed)
    loaders = {}
    for name, df in splits.items():
        tf = train_tf if name == "train" else eval_tf
        loaders[name] = DataLoader(LeafDataset(df, classes, tf), batch_size=batch_size,
                                   shuffle=(name == "train"), generator=gen,
                                   num_workers=workers)
    return loaders


# ---------- Model ----------

def build_model(arch, n_classes):
    """MobileNetV3 with ImageNet weights and a new last layer for our classes."""
    if arch == "small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, n_classes)
    return model


def set_trainable(model, phase):
    """head: only the classifier learns. finetune: also the second half of the features."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if phase == "finetune":
        start = len(model.features) // 2
        for block in model.features[start:]:
            for p in block.parameters():
                p.requires_grad = True


def freeze_batchnorm_stats(model):
    """Frozen layers keep their ImageNet BatchNorm statistics."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def class_weights(labels, n_classes):
    """Rare classes count more in the loss (fixes the 5x imbalance)."""
    counts = np.bincount(labels, minlength=n_classes)
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


# ---------- Training and prediction ----------

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    freeze_batchnorm_stats(model)
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """Softmax probabilities (n_images x n_classes) and true labels."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def fit(model, loaders, classes, args, device):
    """Two-phase training; keeps the weights of the best validation epoch."""
    weights = class_weights(loaders["train"].dataset.labels, len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    names = np.array(classes)
    history, best = [], {"f1": -1.0}
    phases = [("head", args.epochs_head, args.lr_head),
              ("finetune", args.epochs_finetune, args.lr_finetune)]
    epoch = 0
    for phase, n_epochs, lr in phases:
        if n_epochs == 0:
            continue
        set_trainable(model, phase)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=lr, weight_decay=1e-4)
        for _ in range(n_epochs):
            epoch += 1
            start = time.time()
            train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
            probs, y = predict(model, loaders["val"], device)
            m = compute_metrics(names[y], names[probs.argmax(1)], classes)
            history.append({"epoch": epoch, "phase": phase, "train_loss": train_loss,
                            "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
                            "seconds": time.time() - start})
            print(f"  epoch {epoch:2d} [{phase:8s}] loss {train_loss:.3f}  "
                  f"val acc {m['accuracy']:.3f}  val macro-F1 {m['macro_f1']:.3f}  "
                  f"({time.time() - start:.0f}s)")
            if m["macro_f1"] > best["f1"]:
                best = {"f1": m["macro_f1"], "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    print(f"  best epoch: {best['epoch']} (val macro-F1 {best['f1']:.3f})")
    return pd.DataFrame(history), best["epoch"]


# ---------- Saving results ----------

def save_probs(probs, labels, df, classes, out_file):
    """One row per image: path, true class, predicted class, probability per class."""
    table = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
    table.insert(0, "pred", np.array(classes)[probs.argmax(1)])
    table.insert(0, "true", np.array(classes)[labels])
    table.insert(0, "path", df["path"].to_numpy())
    table.to_csv(out_file, index=False)


def plot_history(history, title, out_file):
    """Two small panels: training loss and validation macro-F1 per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, label in [(axes[0], "train_loss", "Training loss"),
                           (axes[1], "val_macro_f1", "Validation macro-F1")]:
        ax.plot(history["epoch"], history[col], color="#2a78d6", linewidth=2, marker="o",
                markersize=4)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("Epoch", color="#52514e")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(color="#e4e3df", linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    switch = history[history["phase"] == "finetune"]["epoch"].min()
    if not np.isnan(switch):
        for ax in axes:
            ax.axvline(switch - 0.5, color="#52514e", linewidth=0.8)
        axes[1].text(switch - 0.4, axes[1].get_ylim()[0], " fine-tune starts",
                     fontsize=8, color="#52514e", va="bottom")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def update_table(out_file, new_rows, key="name"):
    """Replace this run's rows in a CSV that collects all runs."""
    new = pd.DataFrame(new_rows)
    if out_file.exists():
        old = pd.read_csv(out_file)
        new = pd.concat([old[~old[key].isin(new[key])], new])
    new.to_csv(out_file, index=False)
    return new


def write_comparison():
    """Baselines + every (non-quick) CNN run on the test split, in one table."""
    rows = []
    base_file = RESULTS / "baseline_metrics.csv"
    if base_file.exists():
        base = pd.read_csv(base_file)
        rows += base[base["split"] == "test"].rename(columns={"model": "name"}).to_dict("records")
    cnn = pd.read_csv(RESULTS / "cnn_metrics.csv")
    cnn = cnn[(cnn["split"] == "test") & ~cnn["name"].str.endswith("_quick")]
    rows += cnn[["name", "split", "accuracy", "macro_f1"]].to_dict("records")
    table = pd.DataFrame(rows).round(3)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)
    print("\nModel comparison (test split):")
    print(table.to_string(index=False))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run '{args.name}' on {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'no GPU'})")
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    splits = load_splits(args.quick)
    classes = sorted(splits["train"]["class"].unique())
    print(f"Classes ({len(classes)}): {classes}")
    print("Images: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    loaders = make_loaders(splits, classes, args.augment, args.batch_size, device, args.seed)

    model = build_model(args.arch, len(classes)).to(device)
    start = time.time()
    history, best_epoch = fit(model, loaders, classes, args, device)
    train_minutes = (time.time() - start) / 60
    history.to_csv(RESULTS / f"train_history_{args.name}.csv", index=False)
    plot_history(history, f"MobileNetV3-{args.arch}, "
                 f"{'with' if args.augment else 'without'} augmentation",
                 RESULTS / f"train_curve_{args.name}.png")

    names = np.array(classes)
    metric_rows, saved_metrics = [], {}
    for split in ["val", "test"]:
        probs, y = predict(model, loaders[split], device)
        y_true, y_pred = names[y], names[probs.argmax(1)]
        m = compute_metrics(y_true, y_pred, classes)
        saved_metrics[split] = m
        metric_rows.append({"name": f"cnn_{args.name}", "split": split, **m,
                            "best_epoch": best_epoch, "train_minutes": round(train_minutes, 1),
                            "device": device.type})
        save_probs(probs, y, splits[split], classes, RESULTS / f"probs_{args.name}_{split}.csv")
        if split == "test":
            per_class_recall(y_true, y_pred, classes).round(3).rename("recall").to_csv(
                RESULTS / f"cnn_per_class_recall_{args.name}.csv", index_label="class")
            plot_confusion_matrix(y_true, y_pred, classes,
                                  f"MobileNetV3-{args.arch} "
                                  f"({'with' if args.augment else 'no'} augmentation), test split",
                                  RESULTS / f"confusion_cnn_{args.name}_test.png")

    # Everything the app needs to use the model later; threshold is set in Step 5
    torch.save({"state_dict": model.state_dict(), "arch": args.arch, "classes": classes,
                "img_size": IMG_SIZE, "mean": MEAN, "std": STD, "augment": args.augment,
                "best_epoch": best_epoch, "metrics": saved_metrics, "seed": args.seed,
                "threshold": None}, MODELS / f"cnn_{args.name}.pt")

    all_runs = update_table(RESULTS / "cnn_metrics.csv", metric_rows)
    print(f"\nTraining time: {train_minutes:.1f} min on {device.type}")
    print(all_runs.round(3).to_string(index=False))
    write_comparison()
    print(f"\nSaved model to models/cnn_{args.name}.pt")


if __name__ == "__main__":
    main()
