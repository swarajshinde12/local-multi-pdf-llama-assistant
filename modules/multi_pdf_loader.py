import os
import sys
from typing import Dict

# 🔧 Make sure Python can find the project root (C:\local_ai)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modules.pdf_loader import load_pdf_text



def load_multiple_pdfs(folder_path: str) -> Dict[str, str]:
    """
    Loads ALL PDFs inside a folder.
    Returns a dict: {filename: full_text}
    """
    pdf_texts: Dict[str, str] = {}

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder_path, file)
        print(f"📄 Loading PDF: {file}")
        text = load_pdf_text(path)

        if not text.strip():
            print(f"⚠️ {file} has no extractable text, skipping.")
            continue

        pdf_texts[file] = text
        print(f"   ➜ Loaded {len(text)} characters")

    if not pdf_texts:
        print("⚠️ No valid PDFs with text found in this folder.")
    return pdf_texts


if __name__ == "__main__":
    folder = r"C:\local_ai\data"
    pdfs = load_multiple_pdfs(folder)

    print("\nLoaded files:")
    for name, text in pdfs.items():
        print(f"- {name}: {len(text)} chars")
