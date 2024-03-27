from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from langchain.embeddings import HuggingFaceBgeEmbeddings


class BGE_Huggingface_Embeddings(EmbeddingsService):
    def __init__(self, model_path: str, device: str, max_seq_length: Optional[int] = None):
        super().__init__(model_path, device, max_seq_length)
        self.instruction = ""
        if 'zh' in model_path:
            # for chinese model
            self.instruction = "为这个句子生成表示以用于检索相关文章："
        elif 'en' in model_path:
            # for english model
            self.instruction = "Represent this sentence for searching relevant passages:"
        elif 'noinstruct' in model_path:
            # for "bge-large-zh-noinstruct"
            self.instruction = ""
        embedding_model_kwargs = {'device': device}  # cuda or cuda:0
        embedding_encode_kwargs = {'normalize_embeddings': True, 'show_progress_bar': False}
        self.model = HuggingFaceBgeEmbeddings(
            model_name=model_path,
            model_kwargs=embedding_model_kwargs,
            encode_kwargs=embedding_encode_kwargs,
            query_instruction=self.instruction
        )
        self.model.query_instruction = self.instruction

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
