from common.conf import Cfg
from common.utils import RUNTIME_ROOT_DIR
from rerank.reranker_impl import LangchainReranker
from typing import Any, List, Optional

_reranker_model = None
def init_reranker():
    print(RUNTIME_ROOT_DIR)
    cfg = Cfg(RUNTIME_ROOT_DIR + "/conf.toml")
    _reranker_model = LangchainReranker(cfg)

def reranker():
    if _reranker_model is None:
        init_reranker()
    return _reranker_model


def simple_predict(
            query: str,
            passages: List[str],
    ) -> List[str]:
    return reranker().simple_predict(query, passages)