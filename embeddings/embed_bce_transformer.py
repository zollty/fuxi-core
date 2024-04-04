from typing import List, Union, Optional
from .embed_base import EmbeddingsService
from transformers import AutoModel, AutoTokenizer


class BCE_Transformer_Embeddings(EmbeddingsService):
    def __init__(self, model_name: str, model_path: str, device: str, max_seq_length: Optional[int] = None):
        super().__init__(model_name, model_path, device, max_seq_length)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(device)

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

        inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

        # get embeddings
        outputs = self.model(**inputs_on_device, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize

        return embeddings.tolist()
