from sanic_openapi import doc
from sanic_openapi import swagger_blueprint
from sanic import Sanic
from sanic.response import json

app = Sanic(__name__)
app.config.API_VERSION = 'v0.0.1'
app.config.API_TITLE = '异步平台 API文档'
app.blueprint(swagger_blueprint)


@app.get("/class", version=1)
@doc.summary("获取班级信息")  # API信息描述
@doc.tag("班级")  # API分组
@doc.consumes({"class_name": str}, location="query", required=False)
@doc.consumes({"id": int}, location="query", required=True)
async def get_class(request):
    return json({})


@app.get("/studentList", version=1)
@doc.summary("获取学生信息")
@doc.tag("学生")
@doc.consumes({"stu_name": str}, location="query", required=True)
async def get_student(request):
    return json({})


@app.post("/addStudent", version=2)
@doc.summary("新增学生信息")  # API信息描述
@doc.tag("学生")  # API分组
@doc.consumes({"product": {"stu_name": str, "age": int, "city": str}}, location="body")
async def add_student(request):
    return json({})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7500, debug=True)
