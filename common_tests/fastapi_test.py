import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
runtime_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__current_script_path)))
sys.path.append(runtime_root_dir)


from typing import Any, List, Optional, Dict
from fastapi import FastAPI, Body

# from langchain.docstore.document import Document
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

def testpyd22() -> DocumentWithVSId:
    """
    从本地获取configs中配置的embedding模型列表
    """
    data = [DocumentWithVSId(page_content="xxxx", id="xxxx", name="jhdsjhdsjhds", score=2.1),
            DocumentWithVSId(page_content="yyyyyyyy", id="yyy", name="sdds", score=2.1)]
    return data[0]

if __name__ == '__main__':
    app = FastAPI()

    # app.post("/testpyd",
    #          tags=["LLM Management"],
    #          response_model=List[DocumentWithVSId],
    #          summary="zzzzzzzzzzzzzzzzzz"
    #          )(testpyd)
    app.post("/testpyd",
             tags=["LLM Management"],
             response_model=DocumentWithVSId,
             summary="zzzzzzzzzzzzzzzzzz",
             # include_in_schema=False,
             )(testpyd22)

    from fastapi.openapi.utils import get_fields_from_routes
    from pydantic.json_schema import GenerateJsonSchema

    REF_TEMPLATE = "#/components/schemas/{model}"

    schema_generator = GenerateJsonSchema(ref_template=REF_TEMPLATE)

    import json
    print(json.dumps(testpyd(), default=lambda k: k.__dict__))

    print(DocumentWithVSId.schema(
        ref_template=REF_TEMPLATE
    ))

    class Item(BaseModel):
        name: str
        description: str = None
        price: float
        tax: float = None


    class ItemList(BaseModel):
        items: List[Item]

    print(ItemList.schema(
        ref_template=REF_TEMPLATE
    ))

    fields = get_fields_from_routes(list(app.routes or []))
    inputs = [
        (field, field.mode, field._type_adapter.core_schema)
        for field in fields
    ]
    field_mapping, definitions = schema_generator.generate_definitions(
        inputs=inputs
    )

