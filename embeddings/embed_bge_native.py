from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from FlagEmbedding import FlagModel


class BGE_Native_Embeddings(EmbeddingsService):
    def __init__(self, model_path: str, device: str, max_seq_length: Optional[int] = None):
        super().__init__(model_path, device, max_seq_length)
        if 'zh' in model_path:
            # for chinese model
            self.instruction = "为这个句子生成表示以用于检索相关文章："
        elif 'en' in model_path:
            # for english model
            self.instruction = "Represent this sentence for searching relevant passages:"
        elif 'noinstruct' in model_path:
            # for "bge-large-zh-noinstruct"
            self.instruction = ""
        # 不支持设置device，自动检测并自动设置
        self.model = FlagModel(model_path,
                               query_instruction_for_retrieval=self.instruction,
                               use_fp16=True)
        # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def encode(self,
               sentences: Union[str, List[str]],
               to_query: bool = False,
               max_seq_length: Optional[int] = 512,
               batch_size: int = 256,
               show_progress_bar: bool = None,
               device: str = None,
               normalize_embeddings: bool = False,
               query_instruction: str = "",
               ):
        if to_query:
            embeddings = self.model.encode_queries(sentences)
        else:
            embeddings = self.model.encode(sentences)
        return embeddings.tolist()
