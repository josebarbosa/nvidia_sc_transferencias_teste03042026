# file: pdf_to_txt.py
import pdfplumber, pathlib, sys, json

def pdf_para_txt(pdf_path: str, txt_path: str):
    """
    Extrai todo o texto (preservando quebras de linha) de um PDF que contém texto.
    """
    output_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                output_lines.append(txt + "\n")
    all_text = "".join(output_lines)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"✔️  {pdf_path} → {txt_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python pdf_to_txt.py <arquivo.pdf> <saida.txt>")
        sys.exit(1)
    pdf_para_txt(sys.argv[1], sys.argv[2])
