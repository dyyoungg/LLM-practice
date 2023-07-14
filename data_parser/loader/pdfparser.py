from langchain.document_loaders.parsers import PyMuPDFParser, PDFPlumberParser, PyPDFParser
import os


def test_pdf_parser():
    parser = PyPDFParser()
    pages_docs = parser.parse(os.path.join(os.path.dirname(__file__), 'test_file/report.pdf'))
    for page in pages_docs:
        print(page)

from py_pdf_parser import tables
from py_pdf_parser.loaders import load_file
def test_table_extractor():
    import camelot
    tables = camelot.read_pdf(os.path.join(os.path.dirname(__file__), 'test_file/report.pdf'), flavor='stream', pages='all')
    print(tables)
    tables.export('foo.csv', f='csv', compress=True)  # json, excel, html, markdown, sqlite
    for table in tables:
        print(table.parsing_report)


if __name__ == '__main__':
    # test_pdf_parser()
    test_table_extractor()