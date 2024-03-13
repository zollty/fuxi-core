import time
from typing import Any, List, Optional, Dict

import hashlib


def get_short_url(url):
    md5 = hashlib.md5(url.encode('utf-8')).hexdigest()
    # 将32位的md5哈希值转化为10进制数
    num = int(md5, 16)
    # 将10进制数转化为62进制数
    base = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    short_url = ''
    while num > 0:
        short_url += base[num % 62]
        num //= 62
    # 短链接的长度为6位
    return short_url[:6]


# from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.query import VectorQuery

DEFAULT_EMBED_PATH = "/ai/models/bce-embedding-base_v1"
DEFAULT_EMBED_DIMS = 768
REDIS_URL = "redis://127.0.0.1:6389"
EMBED_BATCH_SIZE = 16
DEFAULT_VEC_NUM = 20


def create_schema(kb_name: str, dims: int):
    s = get_short_url(kb_name)
    prefix = "r:" + s
    name = "idx:" + s

    schema = {
        "index": {
            "name": name,
            "prefix": prefix,
        },
        "fields": [
            # {"name": "user", "type": "tag"},
            # {"name": "credit_score", "type": "tag"},
            {"name": "doc", "type": "text"},
            # {"name": "age", "type": "numeric"},
            {
                "name": "embed",
                "type": "vector",
                "attrs": {
                    "dims": dims,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32"
                }
            }
        ]
    }

    return schema


def create_and_run_index(kb_name: str, dims: int = DEFAULT_EMBED_DIMS, redis_url: str = REDIS_URL):
    schema = create_schema(kb_name, dims)
    index = SearchIndex.from_dict(schema)
    # connect to local redis instance
    index.connect(redis_url)

    # create the index (no data yet)
    index.create(overwrite=True)

    return index


def insert_doc(docs: List[str], kb_name: str, redis_url: str = REDIS_URL, embedd_path: str = DEFAULT_EMBED_PATH):
    sentences = docs

    # Embedding a single text
    vectorizer = HFTextVectorizer(model=embedd_path)
    # Embedding a batch of texts
    embeddings = vectorizer.embed_many(sentences, batch_size=EMBED_BATCH_SIZE, as_buffer=True)
    dims = len(embeddings[0])

    schema = create_schema(kb_name, dims)
    index = SearchIndex.from_dict(schema)
    # connect to local redis instance
    index.connect(redis_url)  # "redis://127.0.0.1:6389"

    data = [{"doc": t,
             "embed": v}
            for t, v in zip(sentences, embeddings)]
    # load装载数据
    index.load(data)


def retrieve_docs(query: str, kb_name: str, top_k: int = DEFAULT_VEC_NUM, redis_url: str = REDIS_URL,
                  embedd_path: str = DEFAULT_EMBED_PATH):
    vectorizer = HFTextVectorizer(model=embedd_path)
    # use the HuggingFace vectorizer again to create a query embedding
    query_embedding = vectorizer.embed(query)
    dims = len(query_embedding)

    query = VectorQuery(
        vector=query_embedding,
        vector_field_name="embed",
        return_fields=["doc"],
        num_results=top_k
    )

    schema = create_schema(kb_name, dims)
    index = SearchIndex.from_dict(schema)
    # connect to local redis instance
    index.connect(redis_url)
    results = index.query(query)
    # for x in results:
    #     print(x["doc"], x["vector_distance"])

    return results


if __name__ == '__main__':

    kb_name = "vectorizers"
    create_and_run_index(kb_name)

    sentences = [
        "That is a happy apple",
        "That is a happy person",
        "That is a happy dog",
        "Today is a sunny day"
    ]

    insert_doc(sentences, kb_name)

    results = retrieve_docs("That is a happy cat", kb_name)

    for doc in results:
        print(doc["sentence"], doc["vector_distance"])
