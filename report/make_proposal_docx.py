"""Build the project proposal as a Word file: report/proposal_<surname>.docx

Edit NAME / SURNAME below, then run from the project folder:
    python report/make_proposal_docx.py
"""
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

NAME = "Your Full Name"          # <-- change
SURNAME = "Surname"              # <-- change (file name: proposal_<surname>.docx)
REPO = "https://github.com/thuva1999/Final_Thamilmathy"

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report" / f"proposal_{SURNAME.lower()}.docx"

FONT = "Calibri"
BODY_PT, TABLE_PT = 11, 9
INK, MUTED, ACCENT = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), RGBColor(0x2A, 0x78, 0xD6)
PAGE_W = 21.0 - 2 * 1.9   # A4 width minus 0.75-inch margins (cm)


# ---------- Small helpers ----------

def add_runs(paragraph, text, size=None, color=None):
    """Add text with **bold** and *italic* markers as formatted runs."""
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*)", text):
        if not part:
            continue
        bold, italic = part.startswith("**"), part.startswith("*") and not part.startswith("**")
        run = paragraph.add_run(part.strip("*") if (bold or italic) else part)
        run.bold, run.italic = bold or None, italic or None
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = color
    return paragraph


def new_paragraph(container):
    """A new paragraph; in a fresh table cell, reuse the empty one Word puts there."""
    paragraphs = getattr(container, "paragraphs", [])
    if hasattr(container, "vertical_alignment") and len(paragraphs) == 1 \
            and not paragraphs[0].text and not paragraphs[0].runs:
        return paragraphs[0]
    return container.add_paragraph()


def keep_rows_together(t):
    """Do not split a table row (or the table) across two pages."""
    for row in t.rows:
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True


def fix_widths(t, widths_cm):
    """Make Word respect the column widths (table grid + every cell)."""
    for j, w in enumerate(widths_cm):
        t.columns[j].width = Cm(w)
        for cell in t.columns[j].cells:
            cell.width = Cm(w)
    tbl_pr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)


def para(doc_or_cell, text, size=None, after=3, color=None, align=None):
    p = new_paragraph(doc_or_cell)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(0)
    if align:
        p.alignment = align
    return add_runs(p, text, size, color)


def heading(doc, number, title, points):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{number}. {title} ")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(13), ACCENT
    r2 = p.add_run(f"({points} points)")
    r2.font.size, r2.font.color.rgb = Pt(9), MUTED


def set_cell_borders(cell, top=None, bottom=None):
    """Thin horizontal rules only (cleaner than a full grid)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for side, spec in [("top", top), ("bottom", bottom), ("left", None), ("right", None)]:
        el = OxmlElement(f"w:{side}")
        if spec:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(spec[0]))
            el.set(qn("w:color"), spec[1])
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)
    tc_pr.append(borders)


def table(doc, rows, widths_cm, header=True, size=TABLE_PT):
    """Ruled table: bold header with a dark line, light lines between rows."""
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    t.autofit = False
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Cm(widths_cm[j])
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            add_runs(cell.paragraphs[0], f"**{text}**" if (header and i == 0 and text) else text,
                     size=size)
            is_head = header and i == 0
            set_cell_borders(cell, bottom=(8, "0B0B0B") if is_head else (4, "D8D7D2"))
    fix_widths(t, widths_cm)
    if len(rows) <= 4:          # small tables stay in one piece
        keep_rows_together(t)
    return t


def picture(container, path, width_cm):
    """Image in its own paragraph (container: document or table cell)."""
    p = new_paragraph(container)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    return p


def two_columns(doc, left_width, right_width):
    """Borderless 2-column table for text next to a picture; returns both cells."""
    t = doc.add_table(rows=1, cols=2)
    t.autofit = False
    left, right = t.cell(0, 0), t.cell(0, 1)
    for c in (left, right):
        set_cell_borders(c)
    fix_widths(t, [left_width, right_width])
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))  # keep row on one page
    return left, right


def page_number_footer(section):
    p = section.footer.paragraphs[0]
    add_runs(p, f"CSAI-3210 Project proposal · {NAME} · page ", size=8, color=MUTED)
    run = p.add_run()
    for tag, attr in [("w:fldChar", "begin"), ("w:instrText", None), ("w:fldChar", "end")]:
        el = OxmlElement(tag)
        if attr:
            el.set(qn("w:fldCharType"), attr)
        else:
            el.set(qn("xml:space"), "preserve")
            el.text = "PAGE"
        run._r.append(el)
    run.font.size = Pt(8)


# ---------- Content ----------

def first_rows():
    index = pd.read_csv(ROOT / "data/image_index.csv")
    imgs = [["dataset", "class", "file", "size"]]
    for dataset in ["SAR-CLD-2024", "EWU"]:
        for r in index[index["dataset"] == dataset].head(2).itertuples():
            imgs.append([r.dataset, r._2, Path(r.path).name, f"{r.width}×{r.height}"])
    kb = pd.read_csv(ROOT / "data/knowledge_base.csv")
    kb_rows = [["row", "disease", "feature", "text (start)"]]
    for r in kb[kb["disease"] == "Bacterial Blight"].iloc[[0, 5]].itertuples():
        kb_rows.append([r.row_id, r.disease, r.feature, r.text.replace("\n", "; ")[:58] + "…"])
    return imgs, kb_rows


def build():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name, style.font.size = FONT, Pt(BODY_PT)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    style.paragraph_format.line_spacing = 1.0
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.9)
    sec.top_margin, sec.bottom_margin = Cm(1.6), Cm(1.6)
    page_number_footer(sec)
    imgs, kb_rows = first_rows()

    # Title
    para(doc, "**CottonCare: photo-based cotton leaf check with source-based advice in "
              "Uzbek, Russian and English**", size=15, after=1)
    para(doc, f"{NAME} · CSAI-3210 Introduction to AI · Project proposal · {REPO}",
         size=9, color=MUTED, after=2)

    # 1
    heading(doc, 1, "Problem and user", 15)
    para(doc, "**A cotton farmer in Uzbekistan will use this to photograph a cotton leaf and "
              "learn which disease or problem it has and what to do about it, in Uzbek, Russian "
              "or English.**")
    para(doc, "Cotton is one of Uzbekistan's main crops, and many farms are far from an "
              "agronomist. Leaf curl virus, bacterial blight, jassids (leafhoppers) or leaf "
              "reddening from nutrient and water stress spread fast, so early recognition saves "
              "harvest. Guessing wrong leads to the wrong or an unnecessary pesticide, which "
              "wastes money and harms people and soil, and most treatment information is in "
              "English. CottonCare gives a quick *first check* on a phone, says how sure it is, "
              "and shows cited advice in the farmer's language, so the farmer can decide whether "
              "to act, take a better photo or call an agronomist.")

    # 2
    heading(doc, 2, "Your two AI areas and how they connect", 20)
    side = (PAGE_W - 2.0) / 2
    table(doc, [
        ["", "Area 4: Computer vision (I build and evaluate it)",
         "Area 5: Language models (pretrained, with retrieval)"],
        ["Does", "Classifies a leaf photo into 7 classes (6 problems + healthy) and decides how "
                 "sure it is.", "RAG: finds the knowledge-base rows for the predicted problem "
                                "and turns them into short advice."],
        ["Model", "MobileNetV3-Small (ImageNet weights), fine-tuned: early layers frozen first; "
                  "class-weighted loss.",
         "Exact class→rows mapping, sentence-embedding search (all-MiniLM-L6-v2) as backup; "
         "Google Gemini writes the text."],
        ["In", "Photo, resized to 224×224.", "Predicted class(es), confidence band, language."],
        ["Out", "Top-3 classes + probabilities and a band: **high** (diagnosis), **medium** "
                "(“possibly A or B”), **low** (retake photo); threshold chosen on validation.",
         "What it is, what to do now, how to prevent it, when to call an agronomist, with "
         "source row ids."],
    ], [2.0, side, side])
    para(doc, "**Connection:** photo → **CNN** → class + confidence band → **retrieval** of that "
              "class's rows → **LLM** → advice + source. The CNN output is the retrieval query and "
              "decides what the LLM is asked to do. I build the classifier, the confidence rule, "
              "the retrieval and a safety check; the LLM and embedding model are pretrained. An "
              "LLM answer with any dose or number not in the source is rejected and the source "
              "rows are shown instead (also when there is no API key).", after=2)

    # 3
    heading(doc, 3, "Data", 20)
    table(doc, [
        ["Dataset (Mendeley Data, DOI)", "Size and content", "Use"],
        ["**SAR-CLD-2024**, Daffodil Int. University, Data in Brief 2024 · 10.17632/b3jy2p6k8w",
         "2,137 original field photos (800×800), 7 classes; the 7,000 augmented copies are "
         "**not** used (they would leak test images).", "Train/val/test 70/15/15, stratified, "
                                                        "seed 42"],
        ["**Cotton Leaf Image Dataset**, East West University 2025 · 10.17632/t9hgvk2h9p.1",
         "1,373 original photos (700×700), 5 classes.", "External test: Bacterial Blight, "
                                                         "Healthy Leaf"],
        ["**Cotton disease treatment dataset**, 2025 · 10.17632/533j2mzd4s.1",
         "71 rows: 12 diseases × symptoms, conditions, measures with doses, prevention "
         "(70+ sources).", "RAG knowledge base"],
    ], [5.6, 7.6, PAGE_W - 13.2])
    para(doc, "**First rows** (my image index and the cleaned treatment table):", after=1)
    table(doc, imgs, [2.6, 3.2, 3.4, 1.8], size=8.5)
    para(doc, "", size=4, after=0)
    table(doc, kb_rows, [1.2, 2.6, 3.4, PAGE_W - 7.2], size=8.5)
    para(doc, "", after=0)
    para(doc, "**Problems found.** (1) **Same photo, two labels:** a perceptual-hash check found "
              "293 near-duplicate pairs; 6 SAR-CLD pairs are one photo in two classes (e.g. "
              "CV00018.jpg Curl Virus = HL00085.jpg Healthy Leaf). I remove those 12 photos and keep "
              "all other duplicates in one split (0 pairs cross splits); EWU had 239 extra copies. "
              "(2) **Imbalance:** 578 vs. 116 photos per class (5×), so macro-F1 and class weights. "
              "(3) **Photo style:** SAR-CLD = close-ups on the plant, EWU = picked leaves on paper.")
    left, right = two_columns(doc, PAGE_W - 5.6, 5.6)
    para(left, "**Samples** (one photo per class):", size=9, after=1)
    picture(left, RESULTS / "example_grid.png", PAGE_W - 5.9)
    para(right, "**Plot:** photos per class", size=9, after=1)
    picture(right, RESULTS / "class_counts.png", 5.4)

    # 4
    heading(doc, 4, "How you will know it works", 10)
    para(doc, "**Macro-F1** on the SAR-CLD test split, plus per-class recall and a confusion "
              "matrix: classes are imbalanced and missing a rare problem matters as much as a "
              "common one. **Baselines** on the same split: majority class (simplest) and a random "
              "forest on HSV colour + LBP/GLCM texture features; the CNN with and without "
              "augmentation. **External test:** F1/recall on EWU photos of the shared classes. "
              "**Confidence rule:** share marked “not sure” and accuracy on the rest (coverage "
              "curve). **Advice:** I rate 20 answers 1–5 against the knowledge base, compared "
              "with the raw row; the CSV goes to an agronomist.")
    para(doc, "**First results** (test macro-F1): majority 0.061 · random forest 0.891 · "
              "MobileNetV3-Small 0.958 / 0.957 (without / with augmentation). On EWU photos F1 "
              "drops to 0.16 / 0.41 and the model is often confidently wrong, hence the photo tip, "
              "“not sure” answers and disclaimer.")

    # 5
    heading(doc, 5, "App sketch", 10)
    left, right = two_columns(doc, PAGE_W - 8.4, 8.4)
    para(left, "Streamlit app for a phone browser. The farmer **enters** a language, one leaf "
               "photo (camera or file) and an optional consent to save it. The farmer **gets "
               "back** a result card (class, confidence, runner-up), four-part advice with "
               "source row ids, the source citation and a “first check, not a replacement for "
               "an agronomist” note.")
    para(left, "If unsure, the app says “Possibly A or B” and how to tell them apart; if very "
               "unsure, it asks for a clearer photo. An About page cites all datasets by DOI.")
    picture(right, ROOT / "report/app_sketch.png", 8.2)

    # 6
    heading(doc, 6, "Plan, roles and risks", 15)
    para(doc, "I work alone and do every part. Data checks, splits, baselines, CNN, external "
              "test, confidence rule and advisor already work and are in the repository.", after=2)
    table(doc, [
        ["Week", "Work", "Check-in"],
        ["9", "Data checks, leakage-free splits, baselines, CNN training (CPU, ~10 min/run)",
         "**1:** baseline vs. CNN, confusion matrix"],
        ["10", "External test, confidence thresholds, coverage curve, Grad-CAM", ""],
        ["11", "Knowledge base + RAG advisor (3 languages), safety check, fallback", ""],
        ["12", "Streamlit app end to end", "**2:** live app demo"],
        ["13", "Advice evaluation, translation check, deploy (Streamlit Cloud)", ""],
        ["14", "Robustness (augmentation on Colab GPU), user test, report figures", ""],
        ["15", "Live presentation, final repository", "Final demo"],
    ], [1.2, 11.6, PAGE_W - 12.8])
    para(doc, "", after=0)
    table(doc, [
        ["Biggest risk", "Plan B"],
        ["**1. Fails on real farm photos** (F1 0.41 on EWU)",
         "Confidence bands, photo tip, “not sure” answers; stronger augmentation, more varied "
         "photos, opt-in photos for retraining."],
        ["**2. Free LLM overloaded or wrong**",
         "45-second limit, then source rows shown; dose check; provider can be swapped."],
    ], [5.4, PAGE_W - 5.4])

    # 7
    heading(doc, 7, "Ethics and data rights", 5)
    para(doc, "**Harm:** a farmer trusting a wrong diagnosis may spray the wrong chemical (cost, "
              "health, environment) or miss a spreading disease. Mitigations: “not sure” "
              "answers, advice only from cited sources, no invented doses, an agronomist section, "
              "a visible disclaimer, and a note that some products may not be registered in "
              "Uzbekistan. **Privacy:** no accounts; photos are discarded unless the user opts in; "
              "the LLM never sees the photo. **Data rights:** all three datasets are openly "
              "published on Mendeley Data (licence on each page, expected CC BY 4.0) and cited by "
              "DOI in the README and app; images are not redistributed.")

    # 8
    heading(doc, 8, "Repository and AI-use statement", 5)
    para(doc, f"**Repository:** {REPO} (README with setup and dataset citations; a commit for "
              "every step).")
    para(doc, "**AI use:** I used Claude (Anthropic) in Claude Code as a coding assistant to set "
              "up the project, write and debug code, and help draft and format this proposal; I "
              "checked the results and decisions myself. Google Gemini writes the advice inside "
              "the app.")

    doc.core_properties.title = "CottonCare project proposal"
    doc.core_properties.author = NAME
    doc.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    build()
