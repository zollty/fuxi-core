from langchain.document_loaders import TextLoader
from langchain.embeddings import ModelScopeEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import chardet


# 读取原始文档
#raw_documents_sanguo = TextLoader('/ai/apps/Fuxi-Chatchat/knowledge_base/yby_bgelargezh/content/园博园参考资料.txt', encoding='utf-8').load()
#raw_documents_xiyou = TextLoader('/ai/apps/Fuxi-Chatchat/knowledge_base/yby_bgelargezh/content/园博园介绍.txt', encoding='utf-8').load()
#raw_documents_fw = TextLoader('/ai/apps/Fuxi-Chatchat/knowledge_base/yby_bgelargezh/content/园博园服务.txt', encoding='utf-8').load()
raw_documents_sanguo = TextLoader('/ai/apps/data/园博园参考资料.txt', encoding='utf-8').load()
raw_documents_xiyou = TextLoader('/ai/apps/data/园博园介绍.txt', encoding='utf-8').load()
raw_documents_fw = TextLoader('/ai/apps/data/园博园服务.txt', encoding='utf-8').load()

print(len(raw_documents_sanguo[0].page_content + raw_documents_xiyou[0].page_content + raw_documents_fw[0].page_content))

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents_sanguo = text_splitter.split_documents(raw_documents_sanguo)
documents_xiyou = text_splitter.split_documents(raw_documents_xiyou)
documents_fw = text_splitter.split_documents(raw_documents_fw)
documents = documents_sanguo + documents_xiyou + documents_fw
print("documents nums:", documents.__len__())

# 生成向量（embedding）
# model_id = "/ai/models/nlp_corom_sentence-embedding_chinese-base"
#model_id = "/ai/models/nlp_gte_sentence-embedding_chinese-large"
model_id = "/ai/models/bce-embedding-base_v1"
# embeddings = ModelScopeEmbeddings(model_id=model_id)
model_name = "/ai/models/BAAI_bge-large-zh-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embeddings1 = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
embeddings = HuggingFaceBgeEmbeddings(
    model_name= model_name, #"/ai/models/bce-embedding-base_v1",# "/ai/models/moka-ai_m3e-large", # "/ai/models/BAAI_bge-reranker-large", # "/ai/models/text2vec-bge-large-chinese",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
db = Chroma.from_documents(documents, embedding=embeddings)

# 检索
query1 = "白蛇娘子"
query2 = "美国"
query3 = "照壁"
query = "博纳"
query4 = "梅苑山庄"
query5 = "院融景园"
docs = db.similarity_search(query, k=5)

# 打印结果
for doc in docs:
    print("===================================================")
    print("metadata:", doc.metadata)
    print("page_content:", doc.page_content)