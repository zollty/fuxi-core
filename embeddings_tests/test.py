import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
get_runtime_root_dir = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(get_runtime_root_dir)

from hpdeploy.embeddings import EmbeddingsService, BCE_Sentence_Embeddings, \
    BCE_Huggingface_Embeddings, BCE_Transformer_Embeddings, \
    BGE_Sentence_Embeddings, \
    BGE_Huggingface_Embeddings, BGE_Transformer_Embeddings, \
    Jina_Sentence_Embeddings, Jina_Transformer_Embeddings



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

    # path = "G:/50-TEMP/models/embed/jina-embeddings-v2-base-zh"

    embed = Jina_Sentence_Embeddings(path, device)
    embeddings = embed.encode(sentences)
    print("--------------Jina_Sentence_Embeddings------------------")
    print(embeddings[0][0:10])

    # embed = Jina_Transformer_Embeddings(path, device)
    # embeddings = embed.encode(sentences)
    # print("--------------Jina_Transformer_Embeddings------------------")
    # print(embeddings[0][0:10])

    embed = BGE_Huggingface_Embeddings(path, device)
    embeddings = embed.encode(sentences)
    print("--------------BGE_Huggingface_Embeddings------------------")
    print(embeddings[0][0:10])

    embed = BGE_Transformer_Embeddings(path, device)
    embeddings = embed.encode(sentences)
    print("--------------BGE_Transformer_Embeddings------------------")
    print(embeddings[0][0:10])

