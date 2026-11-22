# CottonCare

A first-check assistant for cotton farmers in Uzbekistan. Take or upload a photo of a cotton leaf. The app says which disease, pest or stress problem the leaf most likely has, and how sure it is. It then gives short, source-based advice in **Uzbek, Russian or English**.

> CottonCare is a first check, not a replacement for an agronomist.

CSAI-3210 Introduction to AI, final project, Webster University in Tashkent (Fall 2026).

## How it works
1. **Computer vision:** a fine-tuned MobileNetV3 (PyTorch, ImageNet weights) classifies the leaf photo and gives its top 3 classes with confidence.
2. **Confidence rule:** a threshold chosen on the validation set decides between a *diagnosis*, a *"possibly A or B"* answer, or *"please take a clearer photo"*.
3. **Language model with retrieval (RAG):** the predicted class looks up rows in a cotton disease treatment table. A pretrained LLM turns those rows into advice in four parts: what it is, what to do now, how to prevent it, and when to call an agronomist. Without an API key, the app shows the table row directly.

## Setup (Windows, PowerShell, CPU only)
```powershell
# 1. Install Python 3.11 (once)
winget install -e --id Python.Python.3.11

# 2. Get the code
git clone https://github.com/thuva1999/Final_Thamilmathy.git
cd Final_Thamilmathy

# 3. Create and activate the virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 4. Install the libraries
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. (Optional) LLM key: copy the example and put your key inside
Copy-Item .env.example .env

# 6. Check everything
python src/check_setup.py
```
If PowerShell blocks `Activate.ps1`, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Datasets (download by hand; images are not in this repo)
Download each dataset from Mendeley Data and unzip it into the folder shown. Keep the folder names inside the zip as they are.

| Dataset | Use in this project | Unzip into |
|---|---|---|
| SAR-CLD-2024 (**Original** images only) | train / validation / test | `data/raw/sarcld2024/` |
| Cotton Leaf Image Dataset, East West University (**Original** set only) | external test | `data/raw/ewu_cotton/` |
| Cotton disease treatment dataset (table) | knowledge base for advice | `data/raw/treatment/` |

Augmented copies are never used. They would leak copies of test images into training.

### Citations
Please cite these datasets if you use this project. Copy the full author list from the "Cite" box on each Mendeley page.

1. *SAR-CLD-2024* (cotton leaf disease dataset). Daffodil International University. Mendeley Data, 2024. DOI: [10.17632/b3jy2p6k8w](https://doi.org/10.17632/b3jy2p6k8w). Described in *Data in Brief*, 2024.
2. *Cotton Leaf Image Dataset for Disease Classification.* East West University. Mendeley Data, 2025. DOI: [10.17632/t9hgvk2h9p.1](https://doi.org/10.17632/t9hgvk2h9p.1).
3. *Cotton disease treatment dataset.* Mendeley Data, 2025. DOI: [10.17632/533j2mzd4s.1](https://doi.org/10.17632/533j2mzd4s.1).

Licences: see each dataset's Mendeley page. These datasets are only downloaded and cited here; they are not redistributed.

## Project structure
```
data/class_mapping.csv   class names shared between the two image datasets
data/splits/             CSV lists of train/val/test file paths (no images)
src/                     all Python code (data checks, training, evaluation, advisor)
app.py                   Streamlit app
models/                  trained model (+ class list and threshold)
results/                 metrics (CSV) and figures (PNG) for the report
```
