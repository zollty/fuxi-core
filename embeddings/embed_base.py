from abc import ABC, abstractmethod
from typing import List, Union, Optional
from fuxi.utils.thread_helper import run_in_executor


class EmbeddingsService(ABC):
    def __init__(self, model_path: str, device: str, max_seq_length: Optional[int] = None):
        self.model_path = model_path
        self.device = device
        self.max_seq_length = max_seq_length

    @abstractmethod
    def encode(self,
               sentences: Union[str, List[str]],
               to_query: bool = False,
               max_seq_length: Optional[int] = None,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               device: str = None,
               normalize_embeddings: bool = False,
               query_instruction: str = "",
               ) -> list[list[float]]:
        """ encode the sentences, return embeddings (List[int]) """

    async def async_encode(self,
                           sentences: Union[str, List[str]],
                           to_query: bool = False,
                           max_seq_length: Optional[int] = None,
                           batch_size: int = 32,
                           show_progress_bar: bool = None,
                           device: str = None,
                           normalize_embeddings: bool = False,
                           query_instruction: str = "",
                           ) -> list[list[float]]:
        return await run_in_executor(None, self.encode, sentences,
                                     to_query=to_query,
                                     max_seq_length=max_seq_length,
                                     device=device,
                                     batch_size=batch_size,
                                     show_progress_bar=show_progress_bar,
                                     normalize_embeddings=normalize_embeddings,
                                     query_instruction=query_instruction)
