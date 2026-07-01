import re
import json
from pathlib import Path
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader=PdfReader(str(pdf_path))
    full_text=""
    for page in reader.pages:
        full_text+=page.extract_text()+"\n"
    return full_text