# Its work is to read the pdf file.

from pypdf import PdfReader


def read_pdf(file_path: str) -> list[tuple[int, str]]:
    # : str - - -- > Type hinting 
    reader = PdfReader(file_path)

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append((i, text))

    return pages