from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from data_parser.loader.pdf_loader import UnstructuredPaddlePDFLoader
from data_parser.spliter.ChineseTextSplitter import ChineseTextSplitter
import os

def load_file(filepath):
    if filepath.lower().endswith('.pdf'):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=300)
        docs = loader.load_and_split(textsplitter)

    write_check_file(filepath, docs)
    print(docs)
    return docs


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


if __name__ == '__main__':
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loader/test_file", "report.pdf")
    docs = load_file(filepath)
    # print(docs)