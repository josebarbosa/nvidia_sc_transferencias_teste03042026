# file: ocr_pdf.py
import os, sys, pathlib
import pdfplumber  # apenas para detectar se o PDF já tem texto
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm

def detect_text_layer(pdf_path: str) -> bool:
    """Retorna True se o PDF já tem camada de texto."""
    with pdfplumber.open(pdf_path) as pdf:
        return any(page.extract_text() for page in pdf.pages)

def ocr_pdf_to_txt(pdf_path: str, txt_path: str, dpi: int = 300):
    # 1) Verifica se já tem texto (e para casos mistos usamos OCR só nas páginas vazias)
    if detect_text_layer(pdf_path):
        # Se já tem texto, usamos pdfplumber direto (mais rápido)
        with pdfplumber.open(pdf_path) as pdf, open(txt_path, "w", encoding="utf-8") as out:
            for page in pdf.pages:
                out.write(page.extract_text() + "\n")
        print(f"✔️  Texto já presente, extraído direto → {txt_path}")
        return

    # 2) Caso contrário, faz OCR página a página
    images = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    with open(txt_path, "w", encoding="utf-8") as out:
        for img in tqdm(images, desc="OCR pages"):
            txt = pytesseract.image_to_string(img, lang="por+eng")  # português + inglês
            out.write(txt + "\n")
    print(f"✔️  OCR concluído → {txt_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python ocr_pdf.py <arquivo.pdf> <saida.txt>")
        sys.exit(1)
    ocr_pdf_to_txt(sys.argv[1], sys.argv[2])
