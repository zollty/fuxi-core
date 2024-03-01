from common.conf import Cfg
from common.utils import RUNTIME_ROOT_DIR
from rerank.reranker_impl import LangchainReranker
from typing import Any, List, Optional


class _Singleton(object):
    reranker_model = None

    def init(self, force: bool = False):
        if force or self.reranker_model is None:
            print(RUNTIME_ROOT_DIR)
            cfg = Cfg(RUNTIME_ROOT_DIR + "/conf.toml")
            self.reranker_model = LangchainReranker(cfg)

    def reranker(self):
        if self.reranker_model is None:
            self.init()
        return self.reranker_model

    def simple_predict(self,
                       query: str,
                       passages: List[str],
                       ) -> List[str]:
        return self.reranker().simple_predict(query, passages)


reranker = _Singleton()
