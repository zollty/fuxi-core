from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from transformers import AutoModel, AutoTokenizer
import torch

class BGE_Transformer_Embeddings(EmbeddingsService):
    def __init__(self, model_path: str, device: str, max_seq_length: Optional[int] = None):
        super().__init__(model_path, device, max_seq_length)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()  # 将模型设置为评估模式
        if 'zh' in model_path:
            # for chinese model
            self.instruction = "为这个句子生成表示以用于检索相关文章："
        elif 'en' in model_path:
            # for english model
            self.instruction = "Represent this sentence for searching relevant passages:"
        elif 'noinstruct' in model_path:
            # for "bge-large-zh-noinstruct"
            self.instruction = ""

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

        if to_query:
            sentences = [self.instruction + q for q in sentences]
        inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")

        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

        # get embeddings
        # outputs = self.model(**inputs_on_device, return_dict=True)
        # embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
        # embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize

        with torch.no_grad():
            model_output = self.model(**inputs_on_device)
            # Perform pooling. In this case, cls pooling.
            embeddings = model_output[0][:, 0]
        # normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()
