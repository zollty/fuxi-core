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


if __name__ == "__main__":
    import sys
    import os

    # 获取当前脚本的绝对路径
    __current_script_path = os.path.abspath(__file__)
    # 将项目根目录添加到sys.path
    RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
    sys.path.append(RUNTIME_ROOT_DIR)


    from common.api_base import ApiRequest
    api = ApiRequest(base_url="http://127.0.0.1:7765")

    passages = [
        "国际园林展区，共设置了如美国休斯敦园、加拿大魁北克园、韩国济州岛园、埃及阿斯旺……等展现欧、亚、非、美和澳洲各国异彩纷呈的园林风格的35个展园。然后，再逐一参观国内各省市自治区的现代园林展区。每个园区的内容，将在具体的参观过程中逐一和您分享。",
        "美国西雅图园，有着翡翠城之花园的称号，展示了水从山峦流动到海湾、绿色植物在夜间发出翡翠色光芒的场景，突出“水维系着翡翠城”的设计主题，彰显翡翠城的自然生态景观和人文景观。",
        "美国的韦恩郡和韦恩斯伯勒园，韦恩斯伯勒园提取美国两位总统华盛顿和麦迪逊的花园环境构建而成，体现出美国园林的规整、简洁和宏大的风格。"]
    query = "有哪些美国园林？"

    data = {
        "passages": passages,
        "query": query,
    }

    response = api.post(
        "/rerank/simple_predict",
        data=data
    )
    ret = api.get_response_value(response, as_json=True)
    print(ret)
