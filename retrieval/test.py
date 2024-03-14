import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

from langchain.document_loaders import TextLoader
from langchain.embeddings import ModelScopeEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import chardet
from retrieval.redis_search import create_and_run_index, insert_doc, retrieve_docs

# def read_file(path, encoding):
#     result = []
#     with open(path, 'r', encoding=encoding) as f:
#         result.append(f.read())
#     return result

# 读取原始文档
raw_documents_sanguo = TextLoader('/ai/apps/data/园博园参考资料.txt', encoding='utf-8').load()
raw_documents_xiyou = TextLoader('/ai/apps/data/园博园介绍.txt', encoding='utf-8').load()
raw_documents_fw = TextLoader('/ai/apps/data/园博园服务.txt', encoding='utf-8').load()

print(
    len(raw_documents_sanguo[0].page_content + raw_documents_xiyou[0].page_content + raw_documents_fw[0].page_content))

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents_sanguo = text_splitter.split_documents(raw_documents_sanguo)
documents_xiyou = text_splitter.split_documents(raw_documents_xiyou)
documents_fw = text_splitter.split_documents(raw_documents_fw)
documents = documents_sanguo + documents_xiyou + documents_fw
print("documents nums:", documents.__len__())
print(documents[0].page_content)
print(documents[:-1].page_content)

if __name__ == '__main__':

    kb_name = "yby"
    create_and_run_index(kb_name)

    docs = [x.page_content for x in documents]

    insert_doc(docs, kb_name)

    sentences = [
        "白蛇娘子",
        "美国",
        "照壁",
        "博纳",
        "梅苑山庄",
        "院融景园",
    ]
    for doc in sentences:
        print(f"-------------------------query: {doc}")
        results = retrieve_docs(doc, kb_name)
        for x in results:
            print(x["doc"], x["vector_distance"])
