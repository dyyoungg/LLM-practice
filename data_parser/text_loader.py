import time

from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from langchain.document_loaders import PDFPlumberLoader, PyPDFLoader, PyMuPDFLoader, PyPDFium2Loader
from data_parser.loader.pdf_loader import UnstructuredPaddlePDFLoader, UnstructuredPdfTextLoader
from data_parser.spliter.ChineseTextSplitter import ChineseTextSplitter
from langchain.text_splitter import SpacyTextSplitter
import os
import requests
import json
import openai

Shangliang_config = {
            'request_url': 'https://lm_experience.sensetime.com/v1/nlp/chat/completions',
            'model_config':
                {
                 "temperature": 0.1,
                 "top_p": 0.7,
                 "max_new_tokens": 2048,
                 "repetition_penalty": 1,
                 "stream": False,
                 "user": "test"}
        }

class OpenAI_Request(object):

    def __init__(self,key, model_name, request_address, generate_config=None):
        super().__init__()
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.model__name = model_name
        self.request_address = request_address
        self.generate_config = generate_config

    def post_request(self, message):

        data = {
            "model": self.model__name,
            "messages":  message
        }

        # add generate parameter of api
        if self.generate_config:
            for k,v in self.generate_config.param_dict.__dict__.items():
                data[k] = v

        data = json.dumps(data)

        response = requests.post(self.request_address, headers=self.headers, data=data)

        return response



def test_pdf_loader(loader_name, filepath, mode='elements', password=None):
    if filepath.lower().endswith('.pdf'):
        if loader_name == 'paddle':
            loader = UnstructuredPdfTextLoader(filepath, mode=mode)

        elif loader_name == 'pymupdf':
            loader = PyMuPDFLoader(filepath, password=password)
        elif loader_name == 'pypdfium2':
            loader = PyPDFium2Loader(filepath, password=password)
        elif loader_name == 'pypdf2':
            loader = UnstructuredPdfTextLoader(filepath, mode='elements')
        elif loader_name == 'pdfplumber':
            loader = PDFPlumberLoader(filepath, password=password)
        docs = loader.load()
        for doc in docs:
            print(doc)
        extractor = TextProcessByLLM()

        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=512)
        # textsplitter = SpacyTextSplitter(pipeline='zh_core_web_sm', chunk_size=512, chunk_overlap=100)
        # extract main text, keyword, table and summarys from each page
        # for page in page_docs:
            # time.sleep(2)
            # page.page_content = textsplitter.split_text(page.page_content)
            # response = extractor.extract_sub_text(page.page_content)
            # # response = extractor.extract_keywords(page.page_content)
            # # response = extractor.extract_table(page.page_content)
            # page.page_content = [response]

        # textsplitter = ChineseTextSplitter(pdf=True, sentence_size=128)
        docs = loader.load_and_split(textsplitter)
    # post processing text information


    write_check_file(filepath, docs, loader_name)

    return docs

class TextProcessByLLM(object):
    def __init__(self):
        self.ShangLiang_cofiguration = Shangliang_config
        openai_keys = "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"
        model_name = "gpt-3.5-turbo"
        request_address = "https://api.openai.com/v1/chat/completions"
        self.openai_requestor = OpenAI_Request(openai_keys, model_name, request_address)

    def get_completion_sl(self, prompt, key="b662fb5453e844499ffdc53547ae7951"):
        url = self.ShangLiang_cofiguration['request_url']
        data = {
            "messages": [{"role": "user", "content": prompt}],
        }
        data.update(self.ShangLiang_cofiguration['model_config'])
        headers = {
            'Content-Type': 'application/json',
            'Authorization': key,
        }
        response = requests.post(url, headers=headers, json=data)
        print(response.status_code)
        if response.status_code == 200:
            res = response.json()['data']['choices'][0]['message']
        else:
            res = ''
        return res

    def get_completion(self, prompt):
        res = self.openai_requestor.post_request(prompt)
        if res.status_code == 200:
            response = res.json()['choices'][0]['message']['content']
            return response
        else:
            status_code = res.status_code
            reason = res.reason
            des = res.text
            print(f'visit error :\n status code: {status_code}\n reason: {reason}\n err description: {des}\n '
                        f'please check whether your account  can access OpenAI API normally')
            return " "
    def get_response(self, prompt):
        success = False
        while success != True:
            try:
                response = self.get_completion_sl(prompt)
                success = True
            except:
                response = self.get_completion_sl(prompt)
                import time
                time.sleep(2)
        return response
    def extract_sub_text(self, content):

        prompt = "Below is a page of an article. Please provide a summary of this page within 80 words while preserving key information as much as possible.\
                  Don't make up information that isn't on this page. Use Chinese. \
                  Here is the paragraph: {}".format(content)
        res = self.get_response(prompt)
        return res

    def extract_keywords(self, content):
        prompt = "Below is a page of an article. \
                 Please extract serveral keywords while preserving key information as much as possible.\
                 Don't make up information that isn't on this page. Use Chinese. \
                 Here is the paragraph: {}".format(content)
        res = self.get_response(prompt)
        return res

    def extract_table(self, content):
        prompt = "You are given a page of content from a document. Your task is to determine whether the page contains tabular data or not.\
                 If tabular data is present, please convert it into a Pandas DataFrame. If no table is present, output one word False.\
                Please note that tabular data refers to information organized in a grid-like structure with columns and rows, commonly seen in spreadsheets or charts.\
                Some characteristics of tabular data include consistent spacing, clear headings, and distinct cells.\
                Here is the paragraph: {}".format(content)

        res = self.get_response(prompt)
        return res



def write_check_file(filepath, docs, loader):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file_'+loader + '.txt')
    with open(fp, 'w', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()

def write_check_file2(filepath, docs, loader):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'text_summary' + loader + '.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        pass


def test_pdfTable_parser(filepath):
    import pdfplumber
    import numpy as np
    import pandas as pd
    if filepath.lower().endswith('.pdf'):
        pdf = pdfplumber.open(filepath)
        print(pdf.metadata)
        first_page = pdf.pages[1]
        table1 = first_page.extract_tables()
        print(np.array(table1).shape)
        # table_df = pd.DataFrame(table1[1:], columns=table1[0])
        print(table1)








if __name__ == '__main__':
    # test different pdf text loader
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loader/test_file", "report.pdf")
    test_loader_name = ['paddle', 'pymupdf', 'pypdfium2', 'pypdf', 'pdfplumber']
    # for loader_name in test_loader_name:
    #     print('test_loader_name',loader_name)
    docs =test_pdf_loader(loader_name='pypdf2', filepath=filepath, mode='paged')

    # filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loader/test_file", 'report_24-25.pdf')
    # test_pdfTable_parser(filepath=filepath)
    # print(docs)

    from textrank4zh import TextRank4Keyword, TextRank4Sentence

    text = '二、本行第十届董事会第六次会议于2023年3月30日审议批准了交通银行股份有限公司2022年度报告及摘要。出席会议应到董事16名, 亲自出席董事15名, 委托出席董事1名, 石磊独立董事因事, 书面委托李晓慧独立董事出席会议并代为行使表决权。'
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    for item in tr4s.get_key_sentences(num=5):
        print(item.index, item.weight, item.sentence)

