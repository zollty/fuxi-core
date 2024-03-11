import uvicorn
from fastapi import FastAPI
import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

from typing import Any, List, Optional, Dict
from fastapi import FastAPI, Body

from langchain.docstore.document import Document
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    data: Any = Field(None, description="API data")


class DocumentWithVSId(BaseModel):
    """
    矢量化后的文档
    """
    id: str = None
    score: float = 3.0

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg."""
        super().__init__(page_content=page_content, **kwargs)

    class Config:
        json_schema_extra = {
            'examples': [
                {
                    'id': 'aaa',
                    'score': 25.0,
                }
            ]
        }


def testpyd() -> List[DocumentWithVSId]:
    """
    从本地获取configs中配置的embedding模型列表
    """
    data = [DocumentWithVSId(page_content="xxxx", id="xxxx", name="jhdsjhdsjhds", score=2.1),
            DocumentWithVSId(page_content="yyyyyyyy", id="yyy", name="sdds", score=2.1)]
    return data


def test2222() -> BaseResponse:
    """
    从本地获取configs中配置的embedding模型列表
    """
    data = [DocumentWithVSId(page_content="xxxx", id="xxxx", name="jhdsjhdsjhds", score=2.1),
            DocumentWithVSId(page_content="yyyyyyyy", id="yyy", name="sdds", score=2.1)]
    return BaseResponse(data=data)


if __name__ == '__main__':
    from common.fastapi_tool import run_api

    app = FastAPI()


    # 添加首页
    @app.get("/")
    def index():
        return "This is Home Page."


    app.get("/testpyd",
            tags=["LLM Management"],
            response_model=List[DocumentWithVSId],
            summary="zzzzzzzzzzzzzzzzzz",
            )(testpyd)

    app.get("/testpyd222",
            tags=["LLM Management"],
            response_model=BaseResponse,
            summary="zzzzzzzzzzzzzzzzzz",
            )(test2222)

    run_api(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="debug",
    )
