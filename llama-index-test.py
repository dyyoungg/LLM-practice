from llama_index import download_loader
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from llama_index import ServiceContext, VectorStoreIndex
from llama_index import SimpleDirectoryReader
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.node_parser import SimpleNodeParser
import os
import openai

service_context = ServiceContext.from_defaults(chunk_size=1000)
# os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = "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"
# openai.api_key = "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"

file_paths = ['./data/text1.txt', './data/text2.txt']
documents = SimpleDirectoryReader(input_files=file_paths).load_data()
# print(documents[1].text)
metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),
        SummaryExtractor(summaries=["prev", "self"]),
        KeywordExtractor(keywords=10),
    ],
)

parser = SimpleNodeParser.from_defaults(
    chunk_size=512,
    include_prev_next_rel=False,
    metadata_extractor=metadata_extractor,
)

service_context = ServiceContext.from_defaults(node_parser=parser)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)