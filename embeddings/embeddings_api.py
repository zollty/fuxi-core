from fastapi import Body
from typing import Dict, List

from fuxi.utils.api_base import (BaseResponse, ListResponse)
from fuxi.utils.runtime_conf import get_log_verbose, logger
from fuxi.utils.thread_cache_pool import ThreadSafeObject, CachePool
from hpdeploy.embeddings.config import OPENAI_EMBEDDINGS_CHUNK_SIZE, get_default_embed_device, \
    get_default_embed_model, get_embed_model_path, get_config_embed_models
from hpdeploy.embeddings import EmbeddingsService, BCE_Native_Embeddings, BCE_Sentence_Embeddings, \
    BCE_Huggingface_Embeddings, BCE_Transformer_Embeddings, BGE_Sentence_Embeddings, \
    Jina_Sentence_Embeddings, Jina_Transformer_Embeddings

DEFAULT_EMBED_SERVICE = {"bce": BCE_Sentence_Embeddings,
                         "bge": BGE_Sentence_Embeddings,
                         "jina": Jina_Sentence_Embeddings,
                         }


class EmbeddingsPool(CachePool):
    def load_embedding(self, model: str, device: str) -> EmbeddingsService:
        self.atomic.acquire()
        key = model
        if not self.get(key):
            item = ThreadSafeObject(key, pool=self)
            self.set(key, item)
            with item.acquire(msg="初始化"):  # for: _pool._cache.move_to_end(self.key)
                item.start_loading()
                self.atomic.release()
                model_path = get_embed_model_path(model)
                if model == "text-embedding-ada-002":  # openai text-embedding-ada-002
                    from langchain.embeddings.openai import OpenAIEmbeddings
                    from hpdeploy.embeddings.langchain_embed import Langchain_Embeddings
                    embeddings = Langchain_Embeddings(OpenAIEmbeddings(model=model,
                                                                       openai_api_key=model_path,
                                                                       chunk_size=OPENAI_EMBEDDINGS_CHUNK_SIZE))
                else:
                    for name, vals in DEFAULT_EMBED_SERVICE.items():
                        if name in model:
                            embeddings = vals(model, model_path, device)
                    if not embeddings:
                        # from langchain.embeddings.huggingface import HuggingFaceEmbeddings
                        # from hpdeploy.embeddings.langchain_embed import Langchain_Embeddings
                        # embeddings = Langchain_Embeddings(HuggingFaceEmbeddings(model_name=get_embed_model_path(model),
                        #                                    model_kwargs={'device': device}))
                        embeddings = BCE_Sentence_Embeddings(model, model_path, device)
                item.obj = embeddings
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(key).obj


embeddings_pool = EmbeddingsPool(cache_num=5)


def load_local_embedding(model: str = None, device: str = None):
    """
    从缓存中加载embeddings，可以避免多线程时竞争加载。
    """
    if model:
        if model not in get_config_embed_models():  # 使用本地Embeddings模型
            raise Exception(f"指定的模型 {model} 不存在")
    else:
        model = get_default_embed_model()
    device = device or get_default_embed_device()
    return embeddings_pool.load_embedding(model=model, device=device)


def embed_texts(
        texts: List[str] = Body(..., description="要嵌入的文本列表", examples=[["hello", "world"]]),
        embed_model: str = Body(None, description=f"使用的嵌入模型。"),
        to_query: bool = Body(False, description="向量是否用于查询。有些模型如Minimax对存储/查询的向量进行了区分优化。"),
) -> BaseResponse:
    """
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    """
    try:
        embeddings = load_local_embedding(model=embed_model)
        data = embeddings.encode(texts, to_query=to_query)
        return BaseResponse(data=data)
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


# 如果是online模型则使用异步线程
async def aembed_texts(
        texts: List[str] = Body(..., description="要嵌入的文本列表", examples=[["hello", "world"]]),
        embed_model: str = Body(None, description=f"使用的嵌入模型。"),
        to_query: bool = Body(False, description="向量是否用于查询。有些模型如Minimax对存储/查询的向量进行了区分优化。"),
) -> BaseResponse:
    """
    see: embed_texts，如果是online模型则使用异步线程
    """
    try:
        embeddings = load_local_embedding(model=embed_model)
        data = await embeddings.async_encode(texts, to_query=to_query)
        print(type(data), data)
        return BaseResponse(data=data)
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")
