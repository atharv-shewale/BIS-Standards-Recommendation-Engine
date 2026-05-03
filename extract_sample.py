import fitz

def extract_sample():
    doc = fitz.open("dataset.pdf")
    text = ""
    for i in range(10, min(15, doc.page_count)):
        text += doc[i].get_text()
    
    with open("sample_pdf_text.txt", "w", encoding="utf-8") as f:
        f.write(text[:5000])

if __name__ == "__main__":
    extract_sample()
