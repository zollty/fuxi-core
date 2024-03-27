from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from langchain.embeddings.base import Embeddings


class Langchain_Embeddings(EmbeddingsService):
    def __init__(self, model: Embeddings):
        self.model = model

    def encode(self,
               sentences: Union[str, List[str]],
               to_query: bool = False,
               max_seq_length: Optional[int] = 512,
               batch_size: int = 512,
               show_progress_bar: bool = None,
               device: str = None,
               normalize_embeddings: bool = False,
               query_instruction: str = "",
               ):
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.model.embed_documents(sentences)
        return embeddings
