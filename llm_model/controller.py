from typing import Any, List, Optional, Dict
import multiprocessing as mp
from fastapi import FastAPI, Body
from common.api_base import (BaseResponse, ListResponse)
from dynaconf import Dynaconf

def mount_controller_routes(app: FastAPI,
                            cfg: Dynaconf,
                            ):
    from fastchat.serve.controller import logger
    from llm_model.shutdown_serve import shutdown_worker_serve, check_worker_processes
    from llm_model.launch_all_serve import launch_worker

    def list_config_models(
            types: List[str] = Body(["local", "online"], description="模型配置项类别，如local, online, worker"),
            placeholder: str = Body(None, description="占位用，无实际效果")
    ) -> BaseResponse:
        '''
        从本地获取configs中配置的模型列表
        '''
        data = {"local": cfg.get("llm.model_cfg", {})}
        return BaseResponse(data=data)

    def start_model(
            model_name: str = Body(None, description="启动该模型"),
    ) -> Dict:
        # return model_worker_ctl(["start_worker", model_name])
        ret, msg = launch_worker(model_name)
        if ret:
            return {"success": True, "code": 200, "msg": "success"}
        else:
            return {"success": False, "code": 501, "msg": msg}

    def stop_model(
            model_name: str = Body(None, description="停止该模型"),
    ) -> Dict:
        # return model_worker_ctl(["stop_worker", model_name])
        if shutdown_worker_serve(model_name):
            return {"success": True, "code": 200, "msg": "success"}
        return {"success": False, "code": 501, "msg": f"the {model_name} worker_processes may not running"}

    def replace_model(
            model_name: str = Body(None, description="停止该模型"),
            new_model_name: str = Body(None, description="替换该模型"),
    ) -> Dict:
        ret = stop_model(model_name)
        if ret.get("success"):
            return start_model(new_model_name)
        else:
            return ret
        # return model_worker_ctl(["replace_worker", model_name, new_model_name])

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

    app.post("/list_config_models",
             tags=["LLM Management"],
             response_model=BaseResponse,
             summary="查看配置的所有模型"
             )(list_config_models)

    return app

