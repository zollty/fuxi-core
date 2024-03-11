from typing import Any, List, Optional, Dict
from fastapi import FastAPI, Body
from common.api_base import (BaseResponse, ListResponse)
from common.utils import decide_device
from dynaconf import Dynaconf
import os

from langchain.docstore.document import Document


class DocumentWithVSId(Document):
    """
    矢量化后的文档
    """
    id: str = None
    score: float = 3.0

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg."""
        super().__init__(page_content=page_content, **kwargs)

    @classmethod
    def __get_validators__(cls):
        #yield cls.validate
        return []

    @classmethod
    def validate(cls, value: Any) -> Any:
        return value


def mount_controller_routes(app: FastAPI,
                            cfg: Dynaconf,
                            ):
    from fastchat.serve.controller import logger
    from llm_model.shutdown_serve import shutdown_worker_serve, check_worker_processes
    from llm_model.launch_all_serve import launch_worker

    def list_llm_models(
            types: List[str] = Body(["local", "online"], description="模型配置项类别，如local, online, worker"),
            placeholder: str = Body(None, description="占位用，无实际效果")
    ) -> BaseResponse:
        """
        从本地获取configs中配置的模型列表
        """
        # 本地模型
        models = cfg.get("llm.model_cfg", {})
        for model in models.values():
            if not model.get("model_path_exists"):
                path = model.get("path", None)
                if path and os.path.isdir(path):
                    model["model_path_exists"] = True
            # device = cfg.get("llm.model_cfg", {}).get(model).get("base", {}).get("device", None)
            # if not device:
            #     device = cfg.get("llm.worker.base.device", "cpu")
            # model["base"]["device"] = decide_device(device)

        data = {"local": cfg.get("llm.model_cfg", {}),
                "online": cfg.get("llm.online_model_cfg", {})}
        return BaseResponse(data=data)

    def list_embed_models() -> BaseResponse:
        """
        从本地获取configs中配置的embedding模型列表
        """
        return BaseResponse(data=cfg.get("llm.embed_model_cfg", {}))

    def testpyd() -> List[DocumentWithVSId]:
        """
        从本地获取configs中配置的embedding模型列表
        """
        data = [DocumentWithVSId(page_content="xxxx", id="xxxx", name="jhdsjhdsjhds"),
                DocumentWithVSId(page_content="yyyyyyyy", id="yyy", name="sdds")]
        return data

    def list_online_embed_models() -> BaseResponse:
        """
        从本地获取configs中配置的online embedding模型列表
        """
        from llm_model import model_workers
        ret = {}
        for k, v in cfg.get("llm.online_model_cfg", {}).items():
            if provider := v.get("provider"):
                worker_class = getattr(model_workers, provider, None)
                if worker_class is not None and worker_class.can_embedding():
                    ret[k] = provider
        return BaseResponse(data=ret)

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
            app._controller.refresh_all_workers()
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

    app.post("/list_llm_models",
             tags=["LLM Management"],
             response_model=BaseResponse,
             summary="查看配置的所有模型"
             )(list_llm_models)

    app.post("/list_embed_models",
             tags=["LLM Management"],
             response_model=BaseResponse,
             summary="查看配置的所有embeddings模型"
             )(list_embed_models)

    app.post("/list_online_embed_models",
             tags=["LLM Management"],
             response_model=BaseResponse,
             summary="查看配置的所有online embeddings模型"
             )(list_online_embed_models)

    app.post("/testpyd",
             tags=["LLM Management"],
             response_model=List[DocumentWithVSId],
             summary="zzzzzzzzzzzzzzzzzz"
             )(testpyd)

    return app
