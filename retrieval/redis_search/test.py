import time


def create_index(vectorizer, sentences, embeddings):
    from redisvl.index import SearchIndex

    # construct a search index from the schema
    index = SearchIndex.from_yaml("/ai/apps/fuxi-core/retrieval/redis_search/hfschema.yaml")

    # connect to local redis instance
    index.connect("redis://127.0.0.1:6389")

    # create the index (no data yet)
    index.create(overwrite=True)

    time.sleep(1)
    print("--------------------1---index.create success")

    # load expects an iterable of dictionaries where
    # the vector is stored as a bytes buffer

    data = [{"sentence": t,
             "embedding": v}
            for t, v in zip(sentences, embeddings)]

    #print(data)

    # load装载数据
    index.load(data)

    time.sleep(1)
    print("--------------------2---index.load success")

    # fetch-取出 by "id"
    john = index.fetch("That is a happy dog")
    print("--------------------3---index.fetch success")
    print(john)

    time.sleep(1)

    print("--------------------4---start to retrieve")
    retrieve(vectorizer, index)


def retrieve(hf, index):
    from redisvl.query import VectorQuery

    # use the HuggingFace vectorizer again to create a query embedding
    query_embedding = hf.embed("That is a happy cat")

    query = VectorQuery(
        vector=query_embedding,
        vector_field_name="embedding",
        return_fields=["sentence"],
        num_results=4
    )

    print("--------------------5---retrieve success")
    results = index.query(query)
    for doc in results:
        print(doc["sentence"], doc["vector_distance"])



if __name__ == '__main__':
    from redisvl.utils.vectorize import HFTextVectorizer

    sentences = [
        "That is a happy apple",
        "That is a happy person",
        "That is a happy dog",
        "Today is a sunny day"
    ]

    # Embedding a single text
    vectorizer = HFTextVectorizer(model="/ai/models/bce-embedding-base_v1")
    embedding = vectorizer.embed("Hello, world!")
    print("Vector dimensions: ", len(embedding))
    print(embedding[:10])

    # Embedding a batch of texts
    embeddings = vectorizer.embed_many(sentences, batch_size=2, as_buffer=True)
    print(embeddings[0][:10])

    create_index(vectorizer, sentences, embeddings)
