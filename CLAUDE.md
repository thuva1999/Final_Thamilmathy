# CottonCare: project brief for Claude

This is my CSAI-3210 Introduction to AI final project at Webster University in Tashkent (Fall 2026). Read this whole file before writing code, and follow it in every session.

## About me and my setup
- Windows laptop, CPU only (no GPU). Use PowerShell commands.
- Python 3.14 in a virtual environment (`.venv`). (Changed from 3.11 on 2026-11-22: 3.14 is already installed.)
- I am not a strong programmer. Explain each step briefly, give exact commands, and say where each file goes.
- Repo: https://github.com/thuva1999/Final_Thamilmathy
- Commit after each working step with a clear message.
- I trained MobileNetV3 before (midterm car brand classifier), so keep a similar style.

## What the app does
A cotton farmer in Uzbekistan takes or uploads a photo of a cotton leaf. The app says which disease, pest or stress problem it most likely has (with confidence), then gives short, source-based advice in Uzbek, Russian or English.

User sentence: "A cotton farmer in Uzbekistan will use this to photograph a cotton leaf and learn which disease or problem it has and what to do about it, in Uzbek, Russian or English."

## Course rules (must follow)
- Combine at least two AI areas, where the output of one feeds the other.
  - Area 4, Computer vision: my leaf classifier (I build, train and evaluate it).
  - Area 5, Language models: pretrained LLM with retrieval (RAG) for advice.
- At least one part built and evaluated by me: the classifier (+ the confidence rule).
- Allowed data: datasets from universities / research institutes. NOT allowed: standard teaching datasets (MNIST, Iris, Titanic, Boston housing, and similar). Do NOT use PlantVillage.
- Must be a deployed app another person can use (Streamlit).
- Regular commits to GitHub.

## Datasets (download manually from Mendeley Data; do NOT commit images to GitHub)
1. **SAR-CLD-2024** (Daffodil International University, Data in Brief 2024) https://data.mendeley.com/datasets/b3jy2p6k8w (DOI 10.17632/b3jy2p6k8w). 2,137 original field photos + 7,000 augmented copies, 7 classes (diseases, pests, stress; includes bacterial blight, curl virus, healthy). USE ONLY THE ORIGINAL IMAGES. Never put augmented copies of a test image in training. Read the real class folder names from the download; don't assume them. → main train / validation / test (e.g., 70/15/15, stratified, seed fixed).
2. **Cotton Leaf Image Dataset for Disease Classification** (East West University, 2025) https://data.mendeley.com/datasets/t9hgvk2h9p/1 (DOI 10.17632/t9hgvk2h9p.1). Original set: Alternaria leaf spot 173, bacterial blight 218, Fusarium wilt 337, healthy 333, Verticillium wilt 312. Use ONLY the original set. → external test on classes shared with SAR-CLD-2024 (at least bacterial blight and healthy). Write a class mapping file `data/class_mapping.csv`.
3. **Cotton disease treatment dataset** (Mendeley Data, 2025) https://data.mendeley.com/datasets/533j2mzd4s/1 (DOI 10.17632/533j2mzd4s.1). Table of 12 cotton diseases/pests with symptoms and treatments from 70+ sources. → knowledge base for RAG. Map each classifier class to its rows.

Folder layout: `data/raw/sarcld2024/`, `data/raw/ewu_cotton/`, `data/raw/treatment/` (all in .gitignore).
Before training: check for duplicate / near-duplicate images (perceptual hash) across splits and report them.

## Pipeline
- Image → resize/crop (224×224) → fine-tuned MobileNetV3-Small or -Large (torchvision, ImageNet weights).
- Output: top-3 classes + softmax confidence.
- Confidence rule (threshold chosen on the validation set, e.g. to keep ≥90% accuracy on accepted images):
  - high → diagnosis + advice;
  - medium → "Possibly A or B" + how to tell them apart (from the knowledge base);
  - low → no diagnosis; ask for a clearer photo of a single leaf.
- RAG: retrieve knowledge-base rows for the predicted class (exact mapping first, sentence-embedding search as backup) → LLM writes advice with four parts: what it is, what to do now, how to prevent it, when to call an agronomist. Show the source row. Never invent pesticide names or doses that are not in the retrieved text. Output language: Uzbek (Latin), Russian or English. Keep the LLM behind one module (`src/advisor.py`) with an API key in `.env`; add a no-LLM fallback that shows the knowledge-base row directly, so the app works without a key.
- Streamlit app: camera/upload, photo preview, result card (class, confidence, runner-up), advice, source, language switch, "first check, not a replacement for an agronomist" note. Works on a phone browser. Uploaded photos are not saved unless the user ticks "allow saving to improve the model".

## Evaluation (for the report)
- Metrics: macro-F1, per-class recall, confusion matrix (SAR-CLD-2024 test split).
- Baselines on the same split:
  - majority class,
  - random forest on colour histograms (HSV) + simple texture features (e.g., LBP or GLCM),
  - fine-tuned MobileNetV3 (main), with and without augmentation.
- External test: F1/recall on the East West University photos for the shared classes → report the drop.
- Confidence rule: share of test images marked "not sure" and accuracy on the rest (coverage vs. accuracy curve).
- Advice: 20 generated answers rated 1–5 by me for correctness against the knowledge base and clarity, vs. showing the raw knowledge-base row. Save a CSV for an agronomist to check.
- Optional: Grad-CAM images to show which part of the leaf the model looks at.
- Save all results as CSV + PNG in `results/`. Training must finish on a CPU in reasonable time (small model, few epochs, freeze early layers first).

## Suggested repo structure
```
Final_Thamilmathy/
  CLAUDE.md
  README.md               # includes dataset citations (DOIs) and download instructions
  requirements.txt
  .env.example            # LLM_API_KEY=
  .gitignore              # .venv, .env, data/raw, models/*.pt, uploads/
  data/class_mapping.csv
  data/splits/            # CSV lists of train/val/test file paths (not the images)
  src/
    explore_data.py       # counts, example grid, image sizes, duplicate check
    make_splits.py
    features.py           # colour + texture features
    baselines.py          # majority + random forest
    train_cnn.py
    evaluate.py           # metrics, confusion matrix, external test, threshold curve
    gradcam.py            # optional
    knowledge_base.py     # load treatment table, map classes, embeddings
    advisor.py            # retrieval + LLM + fallback, 3 languages
  app.py                  # Streamlit
  models/
  results/
```

## Build order (one step per session; stop and let me test after each)
1. Setup: venv, requirements.txt (torch + torchvision CPU, numpy, pandas, pillow, scikit-learn, scikit-image, matplotlib, imagehash, streamlit, sentence-transformers, python-dotenv, LLM client), README, .gitignore, .env.example. Tell me exactly where to unzip each dataset.
2. explore_data.py: class counts per dataset, example grid (one image per class), image sizes, duplicate check (for my proposal's Data section: first rows, one plot, one problem found).
3. make_splits.py + baselines (majority, random forest).
4. CNN training on CPU + evaluation + augmentation comparison.
5. External test + confidence threshold + coverage curve (+ Grad-CAM if time).
6. Knowledge base + advisor (RAG, 3 languages, fallback).
7. Streamlit app.
8. Advice evaluation CSV + all report figures.
9. Deployment (Streamlit Community Cloud or Hugging Face Spaces) with the model file and knowledge base, but not the training images.

## Rules for the code
- Keep it simple and readable; small functions with short comments.
- Fix random seeds; log the class list and threshold with the saved model.
- Never commit .env, API keys, dataset images or user photos.
- Respect dataset licenses: cite every dataset by DOI in README and in the app's About page.
- If something in this brief is unclear (for example the real class names in the download), ask me before guessing.

---

## Progress log and facts found (Claude updates this)
- **2026-11-22, Step 1 (Setup)** done: CLAUDE.md, README.md, requirements.txt, .gitignore, .env.example, src/check_setup.py, git repo connected to GitHub.
- Python 3.14 is used for `.venv` (decided 2026-11-22; no 3.11 install).
- Project folder is inside OneDrive. `.venv` and `data/raw/` are large; OneDrive syncing them can lock files during `pip install`.
- LLM for advice: **Google Gemini free tier** (package `google-genai`, key in `.env` as `LLM_API_KEY`), decided 2026-11-22.
- 2026-11-23: **All 3 datasets in place.** SAR-CLD-2024 originals extracted to `data/raw/sarcld2024/Original Dataset/<class>/*.jpg` (2,137 images, file names like `BBC00001.jpg`). Counts: `Bacterial Blight` 250, `Curl Virus` 431, `Healthy Leaf` 257, `Herbicide Growth Damage` 280, `Leaf Hopper Jassids` 225, `Leaf Redding` 578, `Leaf Variegation` 116. Classes are imbalanced (5× between largest and smallest), so use macro-F1 and class weights.
- **2026-11-23, Step 2 (explore_data.py)** done. Outputs in `results/` (class_counts, example_grid, image_sizes, duplicate_pairs, duplicate_examples, treatment_first_rows, data_summary.txt) and `data/image_index.csv` (path, class, size, pHash per image; reuse in Step 3).
  - All SAR-CLD images are 800×800 JPG; all EWU images are 700×700 PNG.
  - **Problem found (duplicates, pHash distance ≤ 4): 293 pairs, 541 images.** SAR-CLD: 6 cross-class pairs (Curl Virus CV00018–CV00022 = Healthy Leaf HL00085/HL00101–HL00104, distance 0; Leaf Redding LR00064 ≈ Leaf Variegation LV00116) and 6 same-class near-duplicates. EWU: 26 cross-class pairs (mostly Alternaria = Verticillium) and 255 same-class pairs (173 in Healthy Leaf). No duplicates across the two datasets.
  - Step 3 must: keep near-duplicates in the same split (group by hash), and handle the cross-class conflicts (ask the user: drop both copies is the default proposal).
  - Treatment table: 71 non-empty rows; features have typos (`Hebicides`, `Fungisides`, `Preventive_measeures`) to normalise in Step 6. Healthy Leaf has no rows (expected).
- **2026-11-23, Step 3 (make_splits.py, features.py, evaluate.py, baselines.py)** done. User agreed: near-duplicates grouped (same split), cross-class conflict groups removed, EWU deduplicated (one photo per group).
  - SAR-CLD splits (seed 42, stratified by group): train 1,487 / val 319 / test 319; 12 conflict photos removed; 0 near-duplicate pairs across splits.
  - EWU external test: 1,082 unique photos (291 removed), 350 in shared classes (Bacterial Blight 179, Healthy Leaf 171). `data/class_mapping.csv` written.
  - Baselines (test): majority class acc 0.273 / macro-F1 0.061; random forest (54 HSV+LBP+GLCM features, 300 trees, balanced) acc 0.893 / macro-F1 0.891 (val 0.914). The CNN must beat 0.891.
  - `src/evaluate.py` holds shared helpers (compute_metrics, per_class_recall, plot_confusion_matrix). Feature cache: `data/cache/` (gitignored).
- **2026-11-23, Step 4 (in progress):** user chose **Google Colab (T4 GPU)** for training. `src/train_cnn.py` (auto GPU/CPU, `--augment/--no-augment`, `--arch small|large`, `--quick` smoke test; 5 head + 10 fine-tune epochs, class-weighted loss, best epoch by val macro-F1) and `notebooks/train_colab.ipynb` (clones repo, unzips `MyDrive/CottonCare/cottoncare_data.zip` into data/raw, trains both runs, copies models/ + results/ to `MyDrive/CottonCare/step4_output`). Data zip made at `Downloads/cottoncare_data.zip`. CPU smoke test passed (~2 s per 56 images, so a full CPU run would be about 15 min).
- Augmented copies (SAR-CLD and EWU) are never used; they and the extra zips were deleted on 2026-11-23. Only `data/raw/` holds data.
- 2026-11-22: EWU originals extracted to `data/raw/ewu_cotton/Cotton_Original_Dataset/<class>/*.png` (1,373 images). Class folders: `Alternaria Leaf Spot` 173, `Bacterial Blight` 218, `Fusarium Wilt` 337, `Healthy Leaf` 333, `Verticillium Wilt` 312. Shared with SAR-CLD-2024: `Bacterial Blight`, `Healthy Leaf` (same spelling).
- 2026-11-22: Treatment table extracted to `data/raw/treatment/A Comprehensive Dataset of Cotton Plant Diseases f/Data.csv` (82 rows; columns `Disease`, `Feature`, `Description`; blank separator rows between diseases). 12 diseases: Wilt, Bacterial Blight, Herbicide Growth Damage, Leaf Hopper Jassids, Curl Virus, Aphids, Fall Armyworm, Powdery Mildew, Cotton Boll Rot, Leaf Redding, Leaf Variegation, Target spot. All 6 non-healthy SAR-CLD classes match by exact name.
