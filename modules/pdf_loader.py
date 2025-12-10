import os
from typing import List
from pypdf import PdfReader


def load_pdf_text(file_path: str) -> str:
    """
    Read a single PDF file and return its full text as one big string.
    Adds debug checks for corrupted/non-PDF files.
    """
    print(f"🔍 Trying to read PDF: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Check raw header bytes
    with open(file_path, "rb") as f:
        header = f.read(8)
    print(f"📄 File header bytes: {header!r}")

    # Basic PDF validation
    if not header.startswith(b"%PDF-"):
        print("❌ Not a valid PDF — header must start with %PDF-")
        return ""

    # Try loading the PDF
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return ""

    pages_text: List[str] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"⚠️ Error reading page {i}: {e}")
            text = ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)
    print(f"✅ Finished reading PDF. Total characters: {len(full_text)}")
    return full_text


if __name__ == "__main__":
    sample_path = r"C:\local_ai\data\sample.pdf"
    print(f"🔧 Test mode: loading {sample_path}")

    if os.path.exists(sample_path):
        text = load_pdf_text(sample_path)
        if text:
            print("\n📝 First 500 characters:\n")
            print(text[:500])
        else:
            print("⚠️ No extractable text — may be scanned or corrupted.")
    else:
        print("⚠️ sample.pdf not found in /data. Put a real PDF there.")
