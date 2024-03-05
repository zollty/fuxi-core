from fastapi import FastAPI, Body
from typing import Any, List, Optional, Dict
from common.api_base import (BaseResponse, ListResponse)
from common.fastapi_tool import get_httpx_client, HTTPX_DEFAULT_TIMEOUT
from common.utils import LOG_VERBOSE, logger
import time, sys
from common.fastapi_tool import create_app, run_api
import multiprocessing as mp

VERSION = "1.0.0"
LOG_PATH = ""


def mount_controller_routes(app: FastAPI,
                 model_name: str,
                 log_level: str = "INFO",
                 managerQueue: mp.Queue = None,
                 started_event: mp.Event = None,
                 ):
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import logger
    logger.setLevel(log_level)

    from common.fastapi_tool import set_httpx_config

    set_httpx_config()

    # controller = Controller(dispatch_method)
    # sys.modules["fastchat.serve.controller"].controller = controller
    # app._controller = controller

    def start_model(
            new_model_name: str = Body(None, description="释放后加载该模型"),
    ) -> Dict:
        # 与manager进程通信
        managerQueue.put([model_name, "start", new_model_name])
        return {"code": 200, "msg": "done"}

    def stop_model() -> Dict:
        # 与manager进程通信
        managerQueue.put([model_name, "stop", None])
        return {"code": 200, "msg": "done"}

    def replace_model(
            new_model_name: str = Body(None, description="释放后加载该模型"),
    ) -> Dict:
        managerQueue.put([model_name, "replace", new_model_name])
        return {"code": 200, "msg": "done"}

    app.post("/start_worker",
             tags=["LLM Management"],
             response_model=BaseResponse,
             summary="启动"
             )(start_model)

    app.post("/stop_worker",
             tags=["LLM Management"],
             response_model=BaseResponse,
             summary="停止"
             )(stop_model)

    app.post("/replace_worker",
             tags=["LLM Management"],
             response_model=BaseResponse,
             summary="切换"
             )(replace_model)

    return app
    # app.get("/knowledge_base/list_knowledge_bases",
    #         tags=["Knowledge Base Management"],
    #         response_model=ListResponse,
    #         summary="获取知识库列表")(list_kbs)
    #
    # app.post("/chat/knowledge_base_chat",
    #          tags=["Chat"],
    #          summary="与知识库对话")(knowledge_base_chat)


def create_controller_app(
        dispatch_method: str,
        log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import app, Controller, logger
    logger.setLevel(log_level)

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller
    app._controller = controller

    app = create_app([], version=VERSION, title="FastChat Controller")

    mount_controller_routes(app)



    kwargs = get_model_worker_config(model_name)
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address(model_name)
    model_path = kwargs.get("model_path", "")
    kwargs["model_path"] = model_path

    app = create_model_worker_app(log_level=log_level, **kwargs)
    _set_app_event(app, started_event)
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__




    run_api(app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )

    return app