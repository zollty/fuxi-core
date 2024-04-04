from fastapi import Body
from typing import Dict, List

from fuxi.utils.api_base import (BaseResponse, ListResponse)
from fuxi.utils.runtime_conf import get_log_verbose, logger
from fuxi.utils.thread_cache_pool import ThreadSafeObject, CachePool
from hpdeploy.rerank.config import get_default_rerank_device, \
    get_default_rerank_model, get_rerank_model_path, get_config_rerank_models
from hpdeploy.rerank import *

class RerankerPool(CachePool):
    def load_reranker(self, model: str, device: str) -> RerankService:
        self.atomic.acquire()
        key = model
        if not self.get(key):
            item = ThreadSafeObject(key, pool=self)
            self.set(key, item)
            with item.acquire(msg="初始化"):  # for: _pool._cache.move_to_end(self.key)
                item.start_loading()
                self.atomic.release()
                model_path = get_rerank_model_path(model)
                if model.startswith("bge") and model != "bge-reranker-large":
                    reranker = BgeNativeReranker(model, model_path, device)
                else:
                    reranker = SentenceReranker(model, model_path, device)
                item.obj = reranker
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(key).obj


reranker_pool = RerankerPool(cache_num=5)


def load_local_reranker(model: str = None, device: str = None):
    """
    从缓存中加载reranker，可以避免多线程时竞争加载。
    """
    if model:
        if model not in get_config_rerank_models():  # 使用本地reranker模型
            raise Exception(f"指定的模型 {model} 不存在")
    else:
        model = get_default_rerank_model()
    device = device or get_default_rerank_device()
    return reranker_pool.load_reranker(model=model, device=device)


def predict(
        rerank_model: str = Body(None, description=f"使用的Reranker模型。"),
        query: str = Body(False, description="查询字符串"),
        passages: List[str] = Body(..., description="要排序的文本列表", examples=[["hello", "world"]]),
) -> BaseResponse:
    """
    对文本进行重排序检索。返回数据格式：BaseResponse(data=List[float])
    """
    try:
        reranker = load_local_reranker(model=rerank_model)
        data = reranker.predict(query, passages)
        return BaseResponse(data=data)
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本重排序的过程中出现错误：{e}")


# 如果是online模型则使用异步线程
async def apredict(
        rerank_model: str = Body(None, description=f"使用的Reranker模型。"),
        query: str = Body(False, description="查询字符串"),
        passages: List[str] = Body(..., description="要排序的文本列表", examples=[["hello", "world"]]),
) -> BaseResponse:
    """
    see: predict，如果是online模型则使用异步线程
    """
    try:
        reranker = load_local_reranker(model=rerank_model)
        data = await reranker.async_predict(query, passages)
        print(type(data), data)
        return BaseResponse(data=data)
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本重排序的过程中出现错误：{e}")
