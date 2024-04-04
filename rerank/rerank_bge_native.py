from typing import Any, List, Optional
from FlagEmbedding import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker
from .rerank_base import RerankService


class BgeNativeReranker(RerankService):
    """Document compressor that uses `Cohere Rerank API`."""

    def __init__(self, model_name: str, model_path: str, device: str, max_length: int | None = None):
        super().__init__(
            # top_n=top_n,
            model_name=model_name,
            model_path=model_path,
            device=device,
            max_length=max_length,
            # batch_size=batch_size,
            # show_progress_bar=show_progress_bar,
            # num_workers=num_workers,
            # activation_fct=activation_fct,
            # apply_softmax=apply_softmax
        )
        if model_name == "bge-reranker-v2-m3":
            self._model = FlagReranker(model_path, use_fp16=True, max_length=max_length, device=device)
        elif model_name == "bge-reranker-v2-gemma":
            self._model = FlagLLMReranker(model_path, use_fp16=True, max_length=max_length, device=device)
        elif model_name == "bge-reranker-v2-minicpm-layerwise":
            self._model = LayerWiseFlagLLMReranker(model_path, use_fp16=True, max_length=max_length, device=device)
        else:
            raise ValueError(f"model_name: {model_name} not supported")

    def predict(
            self,
            query: str,
            passages: List[str],
    ):
        if len(passages) == 0:  # to avoid empty api call
            return []
        sentence_pairs = [[query, _doc] for _doc in passages]
        if self.model_name == "bge-reranker-v2-minicpm-layerwise":
            return self._model.compute_score(sentence_pairs, cutoff_layers=[28])
        return self._model.compute_score(sentence_pairs)
