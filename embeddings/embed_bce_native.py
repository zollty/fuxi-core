from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from BCEmbedding import EmbeddingModel


class BCE_Native_Embeddings(EmbeddingsService):
    def __init__(self, model_path: str, device: str, max_seq_length: Optional[int] = None):
        super().__init__(model_path, device, max_seq_length)
        self.model = EmbeddingModel(model_name_or_path=model_path, device=device)

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
        if device is None:
            device = self.device
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if not max_seq_length:
            self.model.max_seq_length = max_seq_length
        embeddings = self.model.encode(sentences,
                                       device=device,
                                       batch_size=batch_size,
                                       max_length=max_seq_length,
                                       query_instruction=query_instruction,
                                       )
        return embeddings.tolist()
