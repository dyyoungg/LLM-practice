import os
from pdf2docx import Converter
from pdf2docx import parse

def convert_pdf2docx(pdf_path):
    """
    Convert PDF to DOCX
    """
    cv = Converter(pdf_path)
    docx_path = pdf_path[:-3] + 'docx'
    print(docx_path)
    cv.convert(docx_path, start=0)
    ## extract table
    tables = cv.extract_tables(start=0)
    cv.close()
    for table in tables:
        print(table)




if __name__ == '__main__':
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_file/report.pdf')
    convert_pdf2docx(pdf_path)