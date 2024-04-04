from typing import Any, List, Optional
from sentence_transformers import CrossEncoder
from .rerank_base import RerankService


class SentenceReranker(RerankService):
    """Document compressor that uses `Cohere Rerank API`."""

    def __init__(self, model_path: str, device: str, max_length: int | None = None):
        self._model = CrossEncoder(model_name=model_path, max_length=max_length, device=device)
        super().__init__(
            # top_n=top_n,
            model_path=model_path,
            device=device,
            max_length=max_length,
            # batch_size=batch_size,
            # show_progress_bar=show_progress_bar,
            # num_workers=num_workers,
            # activation_fct=activation_fct,
            # apply_softmax=apply_softmax
        )

    def predict(
            self,
            query: str,
            passages: List[str],
    ):
        if len(passages) == 0:  # to avoid empty api call
            return []
        sentence_pairs = [[query, _doc] for _doc in passages]
        return self._model.predict(sentences=sentence_pairs,
                                   batch_size=self.batch_size,
                                   #  show_progress_bar=self.show_progress_bar,
                                   num_workers=self.num_workers,
                                   #  activation_fct=self.activation_fct,
                                   #  apply_softmax=self.apply_softmax,
                                   convert_to_tensor=False
                                   ).tolist()
