"""Build the 3-page project proposal PDF: report/proposal_<SURNAME>.pdf

Edit NAME / SURNAME below, then run from the project folder:
    python report/make_app_sketch.py      (only if the sketch changed)
    python report/make_proposal.py
"""
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

NAME = "Your Full Name"          # <-- change
SURNAME = "Surname"              # <-- change (file name: proposal_<surname>.pdf)
REPO = "https://github.com/thuva1999/Final_Thamilmathy"

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report" / f"proposal_{SURNAME.lower()}.pdf"

# ---------- Fonts and styles (DejaVu has all the symbols we need) ----------
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

INK, MUTED, ACCENT, RULE = (colors.HexColor(c) for c in
                            ["#0b0b0b", "#52514e", "#2a78d6", "#d8d7d2"])
BODY = ParagraphStyle("body", fontName="DV", fontSize=8.6, leading=11, textColor=INK,
                      alignment=TA_LEFT, spaceAfter=3)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.4, leading=9, textColor=MUTED,
                       spaceAfter=0)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.4, leading=9, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=13.5, leading=16,
                    spaceAfter=2)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=10, leading=12,
                    textColor=ACCENT, spaceBefore=5, spaceAfter=2)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=1.5)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def h(number, title, points):
    return Paragraph(f"{number}. {title} <font color='#52514e' size='7.5'>"
                     f"({points} points)</font>", H2)


def image(path, width):
    """Image scaled to a given width, keeping its shape."""
    w, hgt = PILImage.open(path).size
    return Image(str(path), width=width, height=width * hgt / w)


def table(rows, widths, header=True, font_size=7.4):
    """Simple ruled table; every cell is wrapped as a Paragraph."""
    style = ParagraphStyle("t", parent=CELL, fontSize=font_size, leading=font_size + 1.6)
    head = ParagraphStyle("th", parent=style, fontName="DV-B")
    data = [[Paragraph(str(c), head if (header and i == 0) else style) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK if header else RULE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def side_by_side(left, right, left_width, right_width):
    """Two columns (lists of flowables) without borders."""
    t = Table([[left, right]], colWidths=[left_width, right_width])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (0, -1), 8),
                           ("RIGHTPADDING", (1, 0), (1, -1), 0)]))
    return t


# ---------- Numbers from the project files ----------

def first_rows():
    """First rows of the image index and of the treatment table (for section 3)."""
    index = pd.read_csv(ROOT / "data/image_index.csv")
    imgs = [["dataset", "class", "file", "size"]]
    for dataset in ["SAR-CLD-2024", "EWU"]:
        for r in index[index["dataset"] == dataset].head(2).itertuples():
            imgs.append([r.dataset, r._2, Path(r.path).name, f"{r.width}×{r.height}"])
    kb = pd.read_csv(ROOT / "data/knowledge_base.csv")
    kb_rows = [["row", "disease", "feature", "text (first characters)"]]
    for r in kb[kb["disease"] == "Bacterial Blight"].iloc[[0, 3, 5]].itertuples():
        text = r.text.replace("\n", "; ")
        kb_rows.append([r.row_id, r.disease, r.feature, text[:70] + "…"])
    return imgs, kb_rows


def build():
    W = A4[0] - 2 * 1.5 * cm
    imgs, kb_rows = first_rows()
    story = []

    # ---------- Title ----------
    story += [
        p("CottonCare: photo-based cotton leaf check with source-based advice "
          "in Uzbek, Russian and English", H1),
        p(f"{NAME} · CSAI-3210 Introduction to AI · Project proposal · "
          f"Repository: <link href='{REPO}' color='#2a78d6'>{REPO}</link>", SMALL),
        Spacer(1, 4),
    ]

    # ---------- 1. Problem and user ----------
    story += [
        h(1, "Problem and user", 15),
        p("<b>A cotton farmer in Uzbekistan will use this to photograph a cotton leaf and learn "
          "which disease, pest or stress problem it most likely has and what to do about it, in "
          "Uzbek, Russian or English.</b>"),
        p("Cotton is one of Uzbekistan's main crops, and many farms are far from an agronomist. "
          "Problems such as leaf curl virus, bacterial blight, jassid (leafhopper) attack or leaf "
          "reddening from nutrient and water stress spread quickly; recognising them early "
          "saves part of the harvest. Guessing is costly in the other direction too: the wrong "
          "or unnecessary pesticide wastes money and harms people and soil. Most reliable "
          "treatment information is written in English. CottonCare gives a fast <i>first "
          "check</i> on a phone, says how sure it is, and shows advice that comes from a cited "
          "source, in the farmer's language. It does not replace an agronomist; it helps the "
          "farmer decide whether to act now, take a better photo, or call one."),
    ]

    # ---------- 2. Two AI areas ----------
    story += [
        h(2, "Your two AI areas and how they connect", 20),
        table([
            ["", "Area 4: Computer vision (built and evaluated by me)",
             "Area 5: Language models (pretrained, used with retrieval)"],
            ["What it does", "Classifies a leaf photo into 7 classes (6 problems + healthy) and "
             "decides how sure it is.",
             "Retrieval-augmented generation (RAG): finds the knowledge-base rows for the "
             "predicted problem and turns them into short advice."],
            ["Model", "MobileNetV3-Small (torchvision, ImageNet weights), fine-tuned in two "
             "phases (frozen early layers, then the second half); class-weighted loss.",
             "Retrieval: exact class→rows mapping, sentence-embedding search "
             "(all-MiniLM-L6-v2) as backup. Generation: Google Gemini (free tier)."],
            ["In", "Photo, resized to 224×224.",
             "Predicted class(es), confidence band, chosen language."],
            ["Out", "Top-3 classes with softmax probabilities, and a band: <b>high</b> "
             "(diagnosis), <b>medium</b> (“possibly A or B”), <b>low</b> (ask for a clearer "
             "photo). The high threshold is chosen on the validation set.",
             "Four parts: what it is, what to do now, how to prevent it, when to call an "
             "agronomist — with the ids of the source rows."],
        ], [2.2 * cm, (W - 2.2 * cm) / 2, (W - 2.2 * cm) / 2]),
        Spacer(1, 3),
        p("<b>Connection:</b> photo → <b>CNN</b> → class + confidence band → <b>retrieval</b> "
          "of that class's rows → <b>LLM</b> → advice + source. The CNN output is the "
          "retrieval query and also decides what the LLM is asked to do (full advice, compare "
          "two candidates, or nothing). I build and evaluate the classifier and the confidence "
          "rule myself, and also the retrieval, prompts and a safety check; the LLM and the "
          "embedding model are pretrained. The LLM may only use the retrieved text: an answer "
          "with any number (dose, rate) not found in the source is rejected and the source rows "
          "are shown instead. Without an API key the app shows the rows directly."),
    ]

    # ---------- 3. Data ----------
    story += [
        h(3, "Data", 20),
        table([
            ["Dataset (Mendeley Data)", "Size and content", "Use"],
            ["<b>SAR-CLD-2024</b>, Daffodil International University, <i>Data in Brief</i> "
             "2024 · DOI 10.17632/b3jy2p6k8w",
             "2,137 original field photos (800×800), 7 classes; plus 7,000 augmented copies "
             "that I do <b>not</b> use (they would leak test images into training).",
             "Train / validation / test 70/15/15, stratified, seed 42"],
            ["<b>Cotton Leaf Image Dataset</b>, East West University, 2025 · "
             "DOI 10.17632/t9hgvk2h9p.1",
             "1,373 original photos (700×700), 5 classes; only the original set is used.",
             "External test on the shared classes: Bacterial Blight, Healthy Leaf"],
            ["<b>Cotton disease treatment dataset</b>, 2025 · DOI 10.17632/533j2mzd4s.1",
             "Table of 12 diseases/pests × features (symptoms, conditions, physical, biological "
             "and chemical measures with doses, prevention); 71 rows, from 70+ sources.",
             "Knowledge base for RAG; all 6 problem classes match by name"],
        ], [5.2 * cm, 8.3 * cm, W - 13.5 * cm]),
        Spacer(1, 4),
        p("<b>First rows</b> (image index I built, and the cleaned treatment table):"),
        table(imgs, [2.6 * cm, 3.0 * cm, 3.2 * cm, 1.8 * cm], font_size=7),
        Spacer(1, 2),
        table(kb_rows, [1.1 * cm, 2.6 * cm, 3.0 * cm, W - 6.7 * cm], font_size=7),
        Spacer(1, 4),
    ]

    # Problems as normal text (can flow onto the next page), then pictures side by side
    story += [
        p("<b>Problems found.</b> (1) <b>Duplicates with conflicting labels.</b> A "
          "perceptual-hash check (pHash, ≤ 4 of 64 bits different) found 293 near-duplicate "
          "pairs. In SAR-CLD-2024, 6 pairs are the <i>same photo in two classes</i>, e.g. "
          "CV00018.jpg (Curl Virus) = HL00085.jpg (Healthy Leaf). I remove these 12 photos and "
          "keep every other duplicate group inside one split, so no copy of a test photo is "
          "in training (checked: 0 pairs cross splits). The EWU set had 239 extra copies, "
          "removed for the external test (1,082 unique photos)."),
        p("(2) <b>Class imbalance:</b> Leaf Redding has 578 photos, Leaf Variegation 116 "
          "(5×), so I use macro-F1 and class weights."),
        p("(3) <b>Different photo style:</b> SAR-CLD shows close-ups on the plant, EWU shows "
          "picked leaves on paper — a domain shift the external test will measure."),
    ]
    samples = [p("<b>Samples</b> (one random photo per class):"),
               image(RESULTS / "example_grid.png", W - 6.6 * cm)]
    plot = [p("<b>Plot:</b> photos per class"), image(RESULTS / "class_counts.png", 6.3 * cm)]
    story.append(side_by_side(samples, plot, W - 6.3 * cm, 6.3 * cm))

    # ---------- 4. How you will know it works ----------
    story += [
        h(4, "How you will know it works", 10),
        p("<b>Classifier:</b> <b>macro-F1</b> on the SAR-CLD test split, plus per-class recall "
          "and a confusion matrix. Macro-F1 fits because classes are imbalanced and missing a "
          "rare problem matters as much as missing a common one; accuracy would hide that. "
          "<b>Baselines</b> on the same split: the majority class (the simplest), and a random "
          "forest on HSV colour histograms + LBP/GLCM texture features; the CNN is trained "
          "with and without augmentation. <b>External test:</b> F1/recall on EWU photos of "
          "the shared classes, to report the drop. <b>Confidence rule:</b> share of photos "
          "marked “not sure” and accuracy on the rest (coverage vs. accuracy curve). "
          "<b>Advice:</b> I rate 20 generated answers 1–5 for correctness against the "
          "knowledge base and for clarity, compared with showing the raw row, and save them as "
          "a CSV for an agronomist to check."),
        p("<b>First results</b> (test split, macro-F1): majority class 0.061 · random forest "
          "0.891 · MobileNetV3-Small 0.958 without and 0.957 with augmentation. On EWU photos "
          "of the shared classes F1 drops to 0.16 (no augmentation) / 0.41 (augmentation), and "
          "the model is often confidently wrong — so the app gives a photo tip, a “not sure” "
          "answer and a clear disclaimer."),
    ]

    # ---------- 5. App sketch ----------
    sketch_text = [
        p("Streamlit app that works in a phone browser. The farmer chooses a language, takes or "
          "uploads one leaf photo, and gets back a result card (class, confidence bar, "
          "runner-up), the four-part advice with the ids of the source rows, the full source "
          "citation, and a “first check, not a replacement for an agronomist” note."),
        p("If the model is unsure, the card says “Possibly A or B” and lists how to tell them "
          "apart; if it is very unsure, it asks for a clearer photo instead of guessing. "
          "Photos are not stored unless the user ticks the opt-in box. An About page cites all "
          "three datasets by DOI."),
    ]
    story.append(KeepTogether([
        h(5, "App sketch", 10),
        side_by_side(sketch_text, [image(ROOT / "report/app_sketch.png", 9.0 * cm)],
                     W - 9.0 * cm, 9.0 * cm)]))

    # ---------- 6. Plan, roles, risks ----------
    story.append(KeepTogether([
        h(6, "Plan, roles and risks", 15),
        p("I work alone and do every part. Status: data exploration, splits, baselines, CNN "
          "training, external test, confidence rule and the advisor already work and are in "
          "the repository."),
        table([
            ["Week", "Work", "Shown at check-in"],
            ["9", "Data checks, leakage-free splits, baselines, CNN training (laptop CPU, "
             "~10 min per run)", "<b>Check-in 1:</b> baseline vs. CNN table, confusion matrix"],
            ["10", "External test (EWU), confidence thresholds, coverage curve, Grad-CAM", ""],
            ["11", "Knowledge base + RAG advisor in 3 languages, safety check, no-key fallback",
             ""],
            ["12", "Streamlit app end-to-end (photo → result → advice)",
             "<b>Check-in 2:</b> live demo of the app"],
            ["13", "Advice evaluation (20 answers), translations checked by native speakers, "
             "deployment on Streamlit Community Cloud", ""],
            ["14", "Robustness: stronger augmentation / more epochs on Colab GPU; test with "
             "2–3 users; report figures", ""],
            ["15", "Live presentation; final repository", "Final demo"],
        ], [1.1 * cm, 11.6 * cm, W - 12.7 * cm])]))
    story += [
        Spacer(1, 3),
        table([
            ["Biggest risks", "Plan B"],
            ["<b>1. The model fails on real farm photos</b> (already seen: F1 0.41 on EWU).",
             "Confidence bands + photo tip + “not sure” answers; stronger augmentation and "
             "more varied training photos; opt-in photos for later retraining; report the "
             "limit honestly."],
            ["<b>2. Free LLM is overloaded or writes wrong advice</b> (503/504 errors seen).",
             "Retries with a 45-second limit, then the source rows are shown directly; "
             "safety check against invented doses; the LLM sits behind one module, so the "
             "provider can be changed."],
        ], [6.6 * cm, W - 6.6 * cm]),
    ]

    # ---------- 7. Ethics ----------
    story += [
        h(7, "Ethics and data rights", 5),
        p("<b>Who could be harmed:</b> a farmer who trusts a wrong diagnosis may spray the wrong "
          "or an unnecessary chemical (cost, health, environment) or miss a spreading disease. "
          "Mitigations: confidence bands and “not sure” answers, advice only from cited "
          "sources, no invented doses, an “ask an agronomist” section and a visible "
          "disclaimer. Products in the source come partly from South-Asian studies and may not "
          "be registered in Uzbekistan, so the app says to check local rules. <b>Privacy:</b> "
          "no accounts; photos are processed and discarded unless the user opts in; the LLM "
          "receives only the class name and source text, never the photo. <b>Data rights:</b> "
          "all three datasets are published openly on Mendeley Data by universities and "
          "researchers under the licence shown on each page (I will confirm it is CC BY 4.0); "
          "each is cited by DOI in the README and in the app. Images are not redistributed — "
          "they are kept out of GitHub."),
    ]

    # ---------- 8. Repository ----------
    story += [
        h(8, "Repository and AI-use statement", 5),
        p(f"<b>Repository:</b> <link href='{REPO}' color='#2a78d6'>{REPO}</link> — README with "
          "setup steps and dataset citations, regular commits for every step."),
        p("<b>AI use:</b> I used Claude (Anthropic) through Claude Code as a coding assistant "
          "to set up the project, write and debug the Python code, and help draft and format "
          "this proposal; I checked the results and decisions myself. Google Gemini is used "
          "inside the app to write the advice."),
    ]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            title="CottonCare project proposal", author=NAME)

    def footer(canvas, doc_):
        canvas.setFont("DV", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.5 * cm, 0.65 * cm, f"CSAI-3210 Project proposal · {NAME}")
        canvas.drawRightString(A4[0] - 1.5 * cm, 0.65 * cm, f"Page {doc_.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    build()
