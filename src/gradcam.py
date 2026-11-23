"""Step 5 (optional): Grad-CAM heat maps - which part of the photo the CNN looks at.

Grad-CAM weights the last convolutional feature maps by how much they push the
predicted class up, and shows the result as a heat map over the photo.

Outputs:
  results/gradcam_sar_test.png   one correctly classified SAR-CLD test photo per class
  results/gradcam_ewu.png        EWU photos (shared classes) - where the model fails

Run from the project folder:  python src/gradcam.py [run name, default: main model]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import TEXT_COLOR, choose_main_model, load_checkpoint
from train_cnn import make_transforms

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 42


def grad_cam(model, x, class_idx):
    """Heat map (values 0..1, same size as the input) for one image tensor x (1x3xHxW)."""
    store = {}

    def keep_activation(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    hook = model.features[-1].register_forward_hook(keep_activation)  # last conv block
    model.zero_grad()
    scores = model(x)
    scores[0, class_idx].backward()
    hook.remove()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)       # importance of each map
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    probs = torch.softmax(scores.detach(), 1)[0]
    return (cam / cam.max().clamp(min=1e-8)).detach().numpy(), probs


def explain(model, ckpt, path):
    """Resized photo (as array), heat map, predicted class and its confidence."""
    _, eval_tf = make_transforms(augment=False)
    with Image.open(ROOT / path) as img:
        rgb = img.convert("RGB")
        x = eval_tf(rgb).unsqueeze(0)
        shown = np.array(rgb.resize((ckpt["img_size"], ckpt["img_size"])))
    with torch.no_grad():
        pred = int(model(x).argmax(1))
    cam, probs = grad_cam(model, x, pred)
    return shown, cam, ckpt["classes"][pred], float(probs[pred])


def plot_grid(model, ckpt, items, title, out_file):
    """Two rows per set of photos: the photo, then the photo with its heat map."""
    fig, axes = plt.subplots(2, len(items), figsize=(2.3 * len(items), 5.4), squeeze=False)
    for col, (true_class, path) in enumerate(items):
        shown, cam, pred, conf = explain(model, ckpt, path)
        axes[0, col].imshow(shown)
        axes[0, col].set_title(f"true: {true_class}", fontsize=8, color=TEXT_COLOR)
        axes[1, col].imshow(shown)
        axes[1, col].imshow(cam, cmap="inferno", alpha=0.5)
        axes[1, col].set_title(f"pred: {pred}\n({conf:.0%})", fontsize=8, color=TEXT_COLOR)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else choose_main_model()[0]
    model, ckpt = load_checkpoint(MODELS / f"cnn_{run}.pt")
    print(f"Grad-CAM for run '{run}'")

    test = pd.read_csv(RESULTS / f"probs_{run}_test.csv")
    correct = test[test["true"] == test["pred"]]
    sar_items = [(c, g.sample(1, random_state=SEED)["path"].iloc[0])
                 for c, g in correct.groupby("true")]
    plot_grid(model, ckpt, sar_items, f"Grad-CAM, SAR-CLD test photos ({run})",
              RESULTS / f"gradcam_sar_test_{run}.png")

    ewu = pd.read_csv(RESULTS / f"probs_{run}_ewu.csv")
    shared = ewu[ewu["shared_with_sar"]]
    ewu_items = [(c, p) for c, g in shared.groupby("class")
                 for p in g.sample(3, random_state=SEED)["path"]]
    plot_grid(model, ckpt, ewu_items, f"Grad-CAM, EWU photos ({run})",
              RESULTS / f"gradcam_ewu_{run}.png")
    print(f"Saved results/gradcam_sar_test_{run}.png and results/gradcam_ewu_{run}.png")


if __name__ == "__main__":
    main()
