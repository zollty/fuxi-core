from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from transformers import AutoModel


class Jina_Transformer_Embeddings(EmbeddingsService):
    def __init__(self, model_name: str, model_path: str, device: str, max_seq_length: Optional[int] = None):
        super().__init__(model_name, model_path, device, max_seq_length)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

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
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if max_seq_length:
            embeddings = self.model.encode(sentences, device=device, max_length=max_seq_length)
        else:
            embeddings = self.model.encode(sentences, device=device)
        return embeddings.tolist()
