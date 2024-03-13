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

    data = [{"text": t,
             "embedding": v}
            for t, v in zip(sentences, embeddings)]

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


def retrieve(vectorizer, index):
    from redisvl.query import VectorQuery

    embedding = vectorizer.embed("happy thing")
    print("Vector dimensions: ", len(embedding))
    print(embedding[:10])

    query = VectorQuery(
        vector=embedding,
        vector_field_name="embedding",
        num_results=2
    )
    # run the vector search query against the embedding field
    results = index.query(query)
    print("--------------------5---retrieve success")
    print(results)


if __name__ == '__main__':
    from redisvl.utils.vectorize import HFTextVectorizer

    sentences = [
        "That is a happy dog",
        "That is a happy person",
        "Today is a sunny day"
    ]

    # Embedding a single text
    vectorizer = HFTextVectorizer(model="/ai/models/bce-embedding-base_v1")
    embedding = vectorizer.embed("Hello, world!")
    print("Vector dimensions: ", len(embedding))
    print(embedding[:10])

    # Embedding a batch of texts
    embeddings = vectorizer.embed_many(sentences, batch_size=2)
    print(embeddings[0][:10])

    create_index(vectorizer, sentences, embeddings)
