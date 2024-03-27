from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from sentence_transformers import SentenceTransformer


class Jina_Sentence_Embeddings(EmbeddingsService):
    def __init__(self, model_path: str, device: str, max_seq_length: Optional[int] = None):
        super().__init__(model_path, device, max_seq_length)
        self.model = SentenceTransformer(model_path, trust_remote_code=True, device=device)

    def encode(self,
               sentences: Union[str, List[str]],
               to_query: bool = False,
               max_seq_length: Optional[int] = None,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               device: str = None,
               normalize_embeddings: bool = False,
               query_instruction: str = "",
               ):
        if device is None:
            device = self.device

        embeddings = self.model.encode(sentences,
                                       device=device,
                                       batch_size=batch_size,
                                       show_progress_bar=show_progress_bar,
                                       normalize_embeddings=normalize_embeddings,
                                       )
        return embeddings.tolist()
