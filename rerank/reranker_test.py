import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)


if __name__ == "__main__":
    from langchain.docstore.document import Document
    passages = ["国际园林展区，共设置了如美国休斯敦园、加拿大魁北克园、韩国济州岛园、埃及阿斯旺……等展现欧、亚、非、美和澳洲各国异彩纷呈的园林风格的35个展园。然后，再逐一参观国内各省市自治区的现代园林展区。每个园区的内容，将在具体的参观过程中逐一和您分享。",
                "美国西雅图园，有着翡翠城之花园的称号，展示了水从山峦流动到海湾、绿色植物在夜间发出翡翠色光芒的场景，突出“水维系着翡翠城”的设计主题，彰显翡翠城的自然生态景观和人文景观。",
                "美国的韦恩郡和韦恩斯伯勒园，韦恩斯伯勒园提取美国两位总统华盛顿和麦迪逊的花园环境构建而成，体现出美国园林的规整、简洁和宏大的风格。"]
    docs = [Document(page_content=x) for x in passages]
    query = "有哪些美国园林？"

    from common.conf import Cfg
    from common.utils import RUNTIME_ROOT_DIR
    from rerank.reranker import LangchainReranker

    print(RUNTIME_ROOT_DIR)
    cfg = Cfg(RUNTIME_ROOT_DIR + "/conf.toml")
    reranker_model_path = cfg.get("reranker.model.bge-reranker-large")
    print("-----------------model path------------------")
    print(reranker_model_path)
    reranker_model = LangchainReranker(top_n = cfg.get("reranker.top_n"),
                                    device = (cfg.embedding_device),
                                    max_length = cfg.get("reranker.max_length"),
                                    model_name_or_path = reranker_model_path
                                    )
    print(docs)
    docs = reranker_model.compress_documents(documents=docs,
                                                query=query)
    print("---------after rerank------------------")
    print(docs)

