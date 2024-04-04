from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


class BCE_Huggingface_Embeddings(EmbeddingsService):
    def __init__(self, model_name: str, model_path: str, device: str, max_seq_length: Optional[int] = None):
        super().__init__(model_name, model_path, device, max_seq_length)
        embedding_model_kwargs = {'device': device}  # cuda or cuda:0
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}
        self.model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=embedding_model_kwargs,
            encode_kwargs=embedding_encode_kwargs
        )

    def encode(self,
               sentences: Union[str, List[str]],
               to_query: bool = False,
               max_seq_length: Optional[int] = 512,
               batch_size: int = 32,
               show_progress_bar: bool = False,
               device: str = None,
               normalize_embeddings: bool = True,
               query_instruction: str = "",
               ):
        return self.model.embed_documents(sentences)
