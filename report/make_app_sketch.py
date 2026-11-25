"""Draw the app sketch (main screen on a phone) for the proposal: report/app_sketch.png

Run from the project folder:  python report/make_app_sketch.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "app_sketch.png"
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#c9c8c2"
BLUE, GREEN, AMBER = "#2a78d6", "#1baf7a", "#eda100"


def box(ax, x, y, w, h, face="white", edge=LINE, lw=1.2, r=0.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=face, edgecolor=edge, linewidth=lw))


def text(ax, x, y, s, size=8, color=INK, weight="normal", ha="left"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va="center")


def note(ax, x_text, y_text, x_point, y_point, s, color):
    """Annotation outside the phone with an arrow to the element."""
    ax.annotate(s, (x_point, y_point), (x_text, y_text), fontsize=7.5, color=color,
                va="center", arrowprops=dict(arrowstyle="->", color=color, lw=1))


fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10.4)
ax.axis("off")

# Phone frame
box(ax, 3.2, 0.1, 5.6, 10.2, face="#f4f4f1", edge=INK, lw=1.8, r=0.4)
x0, w = 3.45, 5.1

# Header + language switch
text(ax, x0, 9.85, "CottonCare", 11, weight="bold")
for i, lang in enumerate(["O'zbek", "Русский", "English"]):
    box(ax, x0 + 2.25 + i * 0.95, 9.65, 0.88, 0.4, face=BLUE if i == 0 else "white",
        edge=BLUE)
    text(ax, x0 + 2.69 + i * 0.95, 9.85, lang, 6.5, "white" if i == 0 else BLUE, ha="center")

# Photo tip
box(ax, x0, 9.0, w, 0.5, face="#eef4fc", edge=BLUE)
text(ax, x0 + 0.12, 9.25, "Tip: close-up of ONE leaf on the plant, in daylight", 6.8, BLUE)

# Camera / upload buttons
box(ax, x0, 8.3, 2.45, 0.55, face=BLUE, edge=BLUE)
text(ax, x0 + 1.22, 8.57, "Take photo", 8, "white", "bold", "center")
box(ax, x0 + 2.65, 8.3, 2.45, 0.55, face="white", edge=BLUE)
text(ax, x0 + 3.87, 8.57, "Upload photo", 8, BLUE, "bold", "center")

# Photo preview
box(ax, x0, 6.3, w, 1.85, face="#dfe9d8", edge=LINE)
from matplotlib.patches import Ellipse  # simple leaf: two lobes + a vein
for dx, angle in [(-0.35, 35), (0.35, -35), (0, 0)]:
    ax.add_patch(Ellipse((6.0 + dx, 7.2), 0.75, 1.35, angle=angle, color="#6a9a4c"))
ax.plot([6.0, 6.0], [6.45, 7.8], color="#4d7536", lw=1.2)
text(ax, x0 + 0.1, 6.45, "photo preview", 6.5, MUTED)

# Result card
box(ax, x0, 4.75, w, 1.4, face="white", edge=GREEN, lw=1.6)
text(ax, x0 + 0.15, 5.85, "Leaf reddening", 10, weight="bold")
box(ax, x0 + 3.55, 5.66, 1.4, 0.38, face=GREEN, edge=GREEN)
text(ax, x0 + 4.25, 5.85, "93%  sure", 7.5, "white", "bold", "center")
box(ax, x0 + 0.15, 5.35, 4.8, 0.14, face="#e4e3df", edge="#e4e3df", r=0.07)
box(ax, x0 + 0.15, 5.35, 4.46, 0.14, face=GREEN, edge=GREEN, r=0.07)
text(ax, x0 + 0.15, 5.05, "Runner-up: Leafhoppers (jassids) 4%", 7, MUTED)

# Advice
box(ax, x0, 2.0, w, 2.6, face="white", edge=LINE)
y = 4.35
for title, line in [("What it is", "Red, drying leaf edges; Mg / N shortage, water stress [KB056]"),
                    ("What to do now", "Irrigate if soil is dry; extra nitrogen [KB058]"),
                    ("How to prevent it", "No standing water; resistant varieties [KB060]"),
                    ("When to call an agronomist", "Spreading, no improvement, unsure")]:
    text(ax, x0 + 0.15, y, title, 7.5, weight="bold")
    text(ax, x0 + 0.15, y - 0.27, line, 6.3, MUTED)
    y -= 0.6

# Source + disclaimer + consent
box(ax, x0, 1.3, w, 0.55, face="#f9f9f7", edge=LINE)
text(ax, x0 + 0.12, 1.57, "Source: treatment dataset, DOI 10.17632/533j2mzd4s.1", 6.3, MUTED)
text(ax, x0 + 0.05, 1.0, "First check, not a replacement for an agronomist.", 6.8, AMBER,
     "bold")
box(ax, x0 + 0.05, 0.42, 0.25, 0.25, face="white", edge=INK, r=0.03)
text(ax, x0 + 0.42, 0.55, "Allow saving my photo to improve the model", 6.8)

# Annotations: what the user enters (left) and gets back (right)
note(ax, 9.0, 9.85, x0 + w, 9.85, "ENTERS:\nlanguage", BLUE)
note(ax, 0.05, 8.57, x0, 8.57, "ENTERS:\nleaf photo\n(camera or file)", BLUE)
note(ax, 0.05, 0.55, x0 + 0.05, 0.55, "ENTERS:\nopt-in consent\n(off by default)", BLUE)
note(ax, 9.0, 5.45, x0 + w, 5.45, "GETS BACK:\nclass, confidence,\nrunner-up (CNN)", GREEN)
note(ax, 9.0, 3.3, x0 + w, 3.3, "GETS BACK:\n4-part advice\n(RAG + LLM), with\nsource row ids", GREEN)
note(ax, 9.0, 1.57, x0 + w, 1.57, "GETS BACK:\ncited source", GREEN)

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"Saved {OUT}")
