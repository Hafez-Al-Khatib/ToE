import PyPDF2

try:
    with open("KAEMs.pdf", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for i in range(min(2, len(reader.pages))):
            text += reader.pages[i].extract_text()
        print("KAEMs.pdf Text snippet:")
        print(text[:2000])
except Exception as e:
    print(f"Error reading KAEMs.pdf: {e}")
