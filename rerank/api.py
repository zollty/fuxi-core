from fastapi import FastAPI, Body
from typing import Any, List, Optional
from common.api_base import (BaseResponse, ListResponse)

def mount_reranker_routes(app: FastAPI):
    from rerank.reranker import reranker

    def simple_predict(query: str = Body(..., description="query str"),
                       passages: List[str] = Body(..., description="List[str] to query"),
                       ) -> BaseResponse:
        data = reranker.simple_predict(query, passages)
        return BaseResponse(data=data)

    app.post("/rerank/simple_predict",
             tags=["RAG Rerank"],
             response_model=BaseResponse,
             summary="重排序检索"
             )(simple_predict)

    # app.get("/knowledge_base/list_knowledge_bases",
    #         tags=["Knowledge Base Management"],
    #         response_model=ListResponse,
    #         summary="获取知识库列表")(list_kbs)
    #
    # app.post("/chat/knowledge_base_chat",
    #          tags=["Chat"],
    #          summary="与知识库对话")(knowledge_base_chat)
