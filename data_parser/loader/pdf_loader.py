from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from paddleocr import PaddleOCR
import os
import fitz
import re
import nltk
# from configs.model_config import NLTK_DATA_PATH

# nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

class UnstructuredPaddlePDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load image files, such as PNGs and JPGs."""
    def _get_elements(self) -> List:
        def pdf_ocr_txt(filepath, dir_path="tmp_files"):
            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)
            ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False)
            doc = fitz.open(filepath)
            txt_file_path = os.path.join(full_dir_path, f"{os.path.split(filepath)[-1]}.txt")
            img_name = os.path.join(full_dir_path, 'tmp.png')
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                for i in range(doc.page_count):
                    page = doc[i]
                    text = page.get_text("")
                    fout.write(text)
                    fout.write("\n")
                    img_list = page.get_images()
                    for img in img_list:
                        pix = fitz.Pixmap(doc, img[0])
                        if pix.n - pix.alpha >= 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(img_name)
                        result = ocr.ocr(img_name)
                        ocr_result = [i[1][0] for line in result for i in line]
                        fout.write("\n".join(ocr_result))
            if os.path.exists(img_name):
                os.remove(img_name)
            return txt_file_path

        txt_file_path = pdf_ocr_txt(self.file_path)
        # print('txt_file_path', txt_file_path)
        from unstructured.partition.text import partition_text
        from unstructured.partition.text import element_from_text
        return partition_text(filename=txt_file_path, **self.unstructured_kwargs)



class ChinesePDFTextLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def extract_text_from_pdf(filepath, dir_path="tmp_files"):
            """Extract text content from a PDF file."""
            import PyPDF2
            contents = []
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text().strip()
                    raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
                    new_text = ''
                    for text in raw_text:
                        new_text += text
                        if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                        '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                            contents.append(new_text)
                            new_text = ''
                    if new_text:
                        contents.append(new_text)
            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)
            txt_file_path = os.path.join(full_dir_path, f"{os.path.split(filepath)[-1]}.txt")
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                for text in contents:
                    fout.write(text)
                    fout.write("\n")
                fout.close()
            return txt_file_path

        txt_file_path = extract_text_from_pdf(self.file_path)
        # from unstructured.partition.text import partition_text
        return PartitionChineseText(filename=txt_file_path, **self.unstructured_kwargs)

import jieba
from gensim.models import Word2Vec
def merge_similar_sentences(sentences, window_size=2, threshold=0.8):
    # 分词处理
    seg_sentences = [jieba.lcut(sentence) for sentence in sentences]
    print(seg_sentences)

    # 训练词向量模型
    model = Word2Vec(seg_sentences, min_count=1)

    merged_sentences = []
    for i, sentence in enumerate(sentences):
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        similarity = []
        for j in range(start, end):
            if i != j:
                # 计算关键词的相似度
                if len(seg_sentences[i]) and len(seg_sentences[j]):
                    # 计算关键词的相似度
                    keyword_similarity = model.wv.n_similarity(seg_sentences[i][-1:], seg_sentences[j][:1])
                    similarity.append(keyword_similarity)
                else:
                    similarity.append(0.0)
        if max(similarity) > threshold:
            merged_sentence = sentences[start] + sentences[start + 1]
            merged_sentences.append(merged_sentence)


    return merged_sentences

def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    """Checks if the proportion of non-alpha characters in the text snippet exceeds a given
    threshold. This helps prevent text like "-----------BREAK---------" from being tagged
    as a title or narrative text. The ratio does not count spaces.

    Parameters
    ----------
    text
        The input string to test
    threshold
        If the proportion of non-alpha characters exceeds this threshold, the function
        returns False
    """
    if len(text) == 0:
        return False

    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except:
        return False


def is_possible_title(
        text: str,
        title_max_word_length: int = 20,
        non_alpha_threshold: float = 0.5,
) -> bool:
    """Checks to see if the text passes all of the checks for a valid title.

    Parameters
    ----------
    text
        The input text to check
    title_max_word_length
        The maximum number of words a title can contain
    non_alpha_threshold
        The minimum number of alpha characters the text needs to be considered a title
    """

    # 文本长度为0的话，肯定不是title
    if len(text) == 0:
        print("Not a title. Text is empty.")
        return False

    # 文本中有标点符号，就不是title
    ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    if ENDS_IN_PUNCT_RE.search(text) is not None:
        return False
    # 文本长度不能超过设定值，默认20
    # NOTE(robinson) - splitting on spaces here instead of word tokenizing because it
    # is less expensive and actual tokenization doesn't add much value for the length check
    if len(text) > title_max_word_length:
        return False
    # 文本中数字的占比不能太高，否则不是title
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # NOTE(robinson) - Prevent flagging salutations like "To My Dearest Friends," as titles
    if text.endswith((",", ".", "，", "。")):
        return False

    if text.isnumeric():
        print(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    # 开头的字符内应该有数字，默认5个字符内
    if len(text) < 5:
        text_5 = text
    else:
        text_5 = text[:5]
    alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
    if not alpha_in_text_5:
        return False

    return True


def merge_sentences(sentences):
    """
    :param sentences: sentences list
    :return: merged sentences list

    merge sentences that may cross the page, and merge half sentences between different lines
    """
    merged_sentences = []
    i = 0
    max_sentence_length = 110 # 2 rows
    while i < len(sentences)-1:
        current_sentence = sentences[i]
        next_sentence = sentences[min(i + 1, len(sentences)-1)]
        # 判断开头是否是标题
        possible_have_title = is_possible_title(text=next_sentence.split(" ")[0], title_max_word_length=20,  non_alpha_threshold=0.5)
        # 匹配开头出现的数字
        match = re.match(r"^\d+", next_sentence)
        # 如果第一句话结尾没有结束符号，并且第二局开头是类似于 ”14行长致辞“，前一句为”任德奇“，则这两句可以合并为”任德奇行长致辞“
        if current_sentence[-1] not in ["。","!","?"] and match and (not possible_have_title):
            merge_sentence = current_sentence + next_sentence.replace(match.group(), "")
            merged_sentences.append(merge_sentence)
            i += 2
        # 末尾是[")", "》", "'", "\""] 并且下一句字数少于2行的内容
        elif current_sentence[-1] in [")", "》", "'", "\"",":"]:
            merge_sentence = current_sentence
            while len(next_sentence) < max_sentence_length and merge_sentence[-1] in [")", "》", "'", "\"", ":"]:
                merge_sentence += next_sentence
                i += 1
                next_sentence = sentences[min(i + 1, len(sentences)-1)]
            merged_sentences.append(merge_sentence)
            i += 1
        else:
            merged_sentences.append(current_sentence)
            i += 1
    return merged_sentences






from unstructured.documents.elements import (
    Address,
    Element,
    ElementMetadata,
    ListItem,
    NarrativeText,
    Text,
    Title,
    process_metadata,
)
from typing import IO, Callable, List, Optional
from unstructured.partition.common import exactly_one
from unstructured.file_utils.encoding import read_txt_file
from unstructured.partition.text_type import (
    is_bulleted_text,
    is_possible_narrative_text,
)
from unstructured.cleaners.core import clean_bullets
def element_from_text(text: str) -> Element:
    if is_bulleted_text(text):
        return ListItem(text=clean_bullets(text))
    elif is_possible_narrative_text(text):
        return NarrativeText(text=text)
    elif is_possible_title(text):
        return Title(text=text)
    else:
        return Text(text=text)

def PartitionChineseText(
    filename: Optional[str] = None,
    file: Optional[IO] = None,
    text: Optional[str] = None,
    encoding: Optional[str] = None,
    metadata_filename: Optional[str] = None,
    include_metadata: bool = True,
    **kwargs,
) -> List[Element]:
    if text is not None and text.strip() == "" and not file and not filename:
        return []
    # Verify that only one of the arguments was provided
    exactly_one(filename=filename, file=file, text=text)

    if filename is not None:
        encoding, file_text = read_txt_file(filename=filename, encoding=encoding)

    elif file is not None:
        encoding, file_text = read_txt_file(file=file, encoding=encoding)

    elif text is not None:
        file_text = str(text)

    file_content = file_text.split("\n")
    file_content = merge_sentences(file_content)
    metadata_filename = metadata_filename or filename
    elements: List[Element] = []
    metadata = (
        ElementMetadata(filename=metadata_filename) if include_metadata else ElementMetadata()
    )
    for ctext in file_content:
        ctext = ctext.strip()
        if ctext:
            element = element_from_text(ctext)
            element.metadata = metadata
            elements.append(element)
    return elements






if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    print(os.path.dirname(__file__))
    filepath = './test_file/report.pdf'
    print(os.path.exists(filepath))
    loader = ChinesePDFTextLoader(filepath, mode="elements")
    docs = loader.load()
    print(len(docs))
    for doc in docs:
        print(doc)
    # PartitionChineseText(filename=filepath, encoding="utf-8")
