"""Step 1 check: are Python, the libraries and the dataset folders ready?

Run from the project folder:  python src/check_setup.py
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Library name used in "import ..." for each package in requirements.txt
LIBRARIES = [
    "torch", "torchvision", "numpy", "pandas", "openpyxl", "PIL", "sklearn",
    "skimage", "matplotlib", "imagehash", "streamlit",
    "sentence_transformers", "dotenv", "anthropic",
]

DATA_FOLDERS = ["data/raw/sarcld2024", "data/raw/ewu_cotton", "data/raw/treatment"]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


def check_python():
    """Print the Python version and warn if it is not 3.11."""
    version = sys.version.split()[0]
    print(f"Python {version}  ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        print("  WARNING: the project expects Python 3.11 in .venv")


def check_libraries():
    """Try to import every library and print its version."""
    ok = True
    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
            print(f"  OK       {name} {getattr(module, '__version__', '')}")
        except ImportError:
            print(f"  MISSING  {name}  -> run: pip install -r requirements.txt")
            ok = False
    return ok


def check_data():
    """Count files in each dataset folder (the folders are filled by hand)."""
    for folder in DATA_FOLDERS:
        path = ROOT / folder
        if not path.exists():
            print(f"  MISSING  {folder}/")
            continue
        files = [f for f in path.rglob("*") if f.is_file()]
        images = [f for f in files if f.suffix.lower() in IMAGE_TYPES]
        print(f"  {folder}/: {len(images)} images, {len(files)} files in total")


if __name__ == "__main__":
    print("== Python ==")
    check_python()
    print("== Libraries ==")
    libs_ok = check_libraries()
    if libs_ok:
        import torch
        print(f"  torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("== Datasets ==")
    check_data()
    print("== .env ==")
    print("  found" if (ROOT / ".env").exists() else "  not found (fine: app will use the no-LLM fallback)")
