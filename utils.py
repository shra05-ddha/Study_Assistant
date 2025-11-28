# utils.py

import pdfplumber


# --------------------------
# PDF → TEXT EXTRACTION
# --------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file using pdfplumber.
    Returns a single string containing the entire PDF text.
    """
    final_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                final_text.append(text)
    return "\n\n".join(final_text)


# (Optional) If you want to create reusable prompt strings,
# you can place them here, but we already use templates
# directly in agents.py – this is just for future expansion.

# --------------------------
# HELPER: Clean Text
# --------------------------
def clean_text(text: str) -> str:
    """
    Clean unwanted whitespace.
    """
    return text.replace("\u200b", "").strip()
