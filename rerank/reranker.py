# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import Any, List, Optional
from sentence_transformers import CrossEncoder
from typing import Optional, Sequence
from langchain_core.documents import Document
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import Field, PrivateAttr
from common.conf import Cfg


class LangchainReranker(BaseDocumentCompressor):
    """Document compressor that uses `Cohere Rerank API`."""
    model_name_or_path: str = Field()
    _model: Any = PrivateAttr()
    top_n: int = Field()
    device: str = Field()
    max_length: int = Field()
    batch_size: int = Field()
    # show_progress_bar: bool = None
    num_workers: int = Field()

    # activation_fct = None
    # apply_softmax = False

    def __init__(self, cfg: Cfg):
        top_n: int = cfg.get("reranker.top_n", 3)
        batch_size: int = 32,
        num_workers: int = 0,
        model_name_or_path = cfg.get("reranker.model.bce-reranker-base_v1")
        max_length: int = cfg.get("reranker.max_length", 1024)
        device = cfg.get("embed.device", "cuda")
        # show_progress_bar: bool = None,
        # activation_fct = None,
        # apply_softmax = False,

        self._model = CrossEncoder(model_name=model_name_or_path, max_length=max_length, device=device)
        super().__init__(
            top_n=top_n,
            model_name_or_path=model_name_or_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            # show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            # activation_fct=activation_fct,
            # apply_softmax=apply_softmax
        )

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        sentence_pairs = [[query, _doc] for _doc in _docs]
        results = self._model.predict(sentences=sentence_pairs,
                                      batch_size=self.batch_size,
                                      #  show_progress_bar=self.show_progress_bar,
                                      num_workers=self.num_workers,
                                      #  activation_fct=self.activation_fct,
                                      #  apply_softmax=self.apply_softmax,
                                      convert_to_tensor=True
                                      )
        top_k = self.top_n if self.top_n < len(results) else len(results)

        values, indices = results.topk(top_k)
        final_results = []
        for value, index in zip(values, indices):
            doc = doc_list[index]
            doc.metadata["relevance_score"] = value
            final_results.append(doc)
        return final_results

