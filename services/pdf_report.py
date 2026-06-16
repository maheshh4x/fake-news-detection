from fpdf import FPDF
import re

def clean_pdf_text(text: str):
    """
    Remove emojis / unsupported unicode characters for PDF Helvetica font.
    """
    if text is None:
        return ""
    text = str(text)

    # remove emojis and symbols
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # keep only ASCII
    return text

def generate_pdf_report(data: dict, output_path: str):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, text="Fake News Detection Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    pdf.set_font("Helvetica", size=11)

    for key, value in data.items():
        key_clean = clean_pdf_text(key)
        value_clean = clean_pdf_text(value)
        pdf.multi_cell(0, 8, text=f"{key_clean}: {value_clean}", new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    return output_path

