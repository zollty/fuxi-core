from sanic import Sanic
from sanic.response import ResponseStream
from sanic.response import json as sanic_json
from sanic.response import text as sanic_text
from sanic_ext import openapi
import json

from typing import Any, List, Optional, Dict

import datetime, decimal
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        # elif isinstance(obj, array):
        #    return obj.tolist()
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        else:
            return super(MyEncoder, self).default(obj)


# from langchain.docstore.document import Document
# from pydantic import BaseModel, Field

class DocumentWithVSId():
    """
    矢量化后的文档
    """
    id: str = None
    score: float = 3.0

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg."""
        # super().__init__(page_content=page_content, **kwargs)

    class Config:
        schema_extra = {
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


def mount_app(app):
    @app.route("/testpyd", name="testpyd")
    async def test(request):
        return sanic_json(json.dumps(testpyd(), cls=MyEncoder, ensure_ascii=False))  # , default=lambda k: k.__dict__

    @app.route("/", name="index")
    async def test(request):
        return sanic_json({"hello": "world"})

    @app.get("/class", version=1, name="class")
    @openapi.summary("获取班级信息")  # API信息描述
    @openapi.tag("班级")  # API分组
    @openapi.parameter({"class_name": str}, location="query", required=False)
    @openapi.parameter({"id": int}, location="query", required=True)
    async def get_class(request):
        return sanic_json({})

    @app.get("/studentList", version=1, name="studentList")
    @openapi.summary("获取学生信息")
    @openapi.tag("学生")
    @openapi.parameter({"stu_name": str}, location="query", required=True)
    async def get_student(request):
        return sanic_json({})

    @app.post("/addStudent", version=2, name="addStudent")
    @openapi.summary("新增学生信息")  # API信息描述
    @openapi.tag("学生")  # API分组
    @openapi.parameter({"product": {"stu_name": str, "age": int, "city": str}}, location="body")
    async def add_student(request):
        return sanic_json({})


appp = Sanic("QAnything")
appp.config.API_VERSION = 'v0.0.1'
appp.config.API_TITLE = '异步平台 API文档'
# 设置请求体最大为 400MB
appp.config.REQUEST_MAX_SIZE = 400 * 1024 * 1024
mount_app(appp)

if __name__ == "__main__":
    appp.run(host='0.0.0.0', port=7500, workers=10, debug=True)
