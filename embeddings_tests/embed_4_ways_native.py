import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
get_runtime_root_dir = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(get_runtime_root_dir)


def embed_interface(path, device, sentences):
    from transformers import AutoModel, AutoTokenizer

    # init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True)

    model.to(device)

    # get inputs
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    # get embeddings
    outputs = model(**inputs_on_device, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize: <class 'torch.Tensor'>

    start_time = time.time()
    for i in range(tn):
        # extract embeddings
        # get inputs
        inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

        # get embeddings
        outputs = model(**inputs_on_device, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize

    end_time = time.time() - start_time
    print(type(embeddings))
    print(end_time)
    print(embeddings.tolist()[0][0:10])

    from sentence_transformers import SentenceTransformer

    # init embedding model
    ## New update for sentence-trnasformers. So clean up your "`SENTENCE_TRANSFORMERS_HOME`/maidalun1020_bce-embedding-base_v1" or "～/.cache/torch/sentence_transformers/maidalun1020_bce-embedding-base_v1" first for downloading new version.
    model = SentenceTransformer(path, device=device)

    # extract embeddings
    embeddings = model.encode(sentences, normalize_embeddings=True)  # <class 'numpy.ndarray'>

    start_time = time.time()
    for i in range(tn):
        # extract embeddings
        embeddings = model.encode(sentences, normalize_embeddings=True)

    end_time = time.time() - start_time
    print(type(embeddings))
    print(end_time)
    print(embeddings.tolist()[0][0:10])

    from langchain.embeddings import HuggingFaceEmbeddings

    # init embedding model
    embedding_model_name = path
    embedding_model_kwargs = {'device': device}  # cuda or cuda:0
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )

    embeddings = embed_model.embed_documents(sentences)

    start_time = time.time()
    for i in range(tn):
        # extract embeddings
        embeddings = embed_model.embed_documents(sentences)

    end_time = time.time() - start_time
    print(end_time)
    print(embeddings[0][0:10])


def embed_bce(path, device, sentences):
    # init embedding model
    model = EmbeddingModel(model_name_or_path=path, device=device)
    embeddings = model.encode(sentences)  # <class 'numpy.ndarray'>
    print(embeddings[0][0:10])

    start_time = time.time()
    for i in range(tn):
        # extract embeddings
        embeddings = model.encode(sentences)

    end_time = time.time() - start_time
    print(end_time)
    print(type(embeddings))
    print(embeddings.tolist()[0][0:10])


if __name__ == "__main__":
    import platform
    from BCEmbedding import EmbeddingModel
    import time
    from fuxi.utils.torch_helper import detect_device

    os_type = platform.system()
    print(os_type)
    if os_type == "Windows":
        path = "G:/50-TEMP/models/embed/bce-embedding-base_v1"
    else:
        path = "/ai/models/bce-embedding-base_v1"

    print(detect_device())
    tn: int = 100
    device = 'cuda:0'  # if no GPU, set "cpu"

    # list of sentences
    sentences = [
        '我是AI智能机器人，你可以称呼我为聊天助手或者知识顾问。我是由邹天涌先生所指导的，负责回答和解决各种问题，提供信息和帮助。如果你有任何问题，尽管问吧！',
        'sentence_1']

    embed_bce(path, device, sentences)
    embed_interface(path, device, sentences)
    path = "G:/50-TEMP/models/embed/jina-embeddings-v2-base-zh"

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(path, trust_remote_code=True, device=device)
    embeddings = model.encode(sentences, device=device)

    print(type(embeddings))
    print(embeddings.tolist()[0][0:10])

    from transformers import AutoModel
    from numpy.linalg import norm

    cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))
    # trust_remote_code is needed to use the encode method
    model = AutoModel.from_pretrained(path, trust_remote_code=True)
    print(type(model))  # transformers_modules.jina-embeddings-v2-base-zh.modeling_bert.JinaBertModel
    embeddings = model.encode(sentences, device=device)
    print(type(embeddings))
    print(embeddings.tolist()[0][0:10])
