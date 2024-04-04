from typing import Any, List, Optional
from abc import ABC, abstractmethod
from fuxi.utils.thread_helper import run_in_executor


class RerankService(ABC):
    """Document compressor that uses `Cohere Rerank API`."""

    def __init__(self, model_name: str, model_path: str, device: str, max_length: int | None = None):
        from hpdeploy.rerank import config
        self.model_name = model_name
        self.batch_size: int = config.get_default_rerank_batch_size()
        self.num_workers: int = config.get_default_rerank_num_workers()
        self.model_path = model_path
        self.max_length = max_length
        self.device = device
        # show_progress_bar: bool = None,
        # activation_fct = None,
        # apply_softmax = False,

    @abstractmethod
    def predict(
            self,
            query: str,
            passages: List[str],
    ) -> list[float]:
        """ interface to rerank model """

    async def async_predict(self,
                            query: str,
                            passages: List[str],
                            ) -> list[float]:
        return await run_in_executor(None, self.predict, query, passages)
