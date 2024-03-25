from typing import Any, List, Optional, Dict
from fastapi import FastAPI, Body
from fuxi.utils.api_base import (BaseResponse, ListResponse)
from dynaconf import Dynaconf
import os
import threading

global_worker_dict = {}


def find_use_port():
    start_port = 21105
    ports = [x["port"] for _, x in global_worker_dict.items()]
    while True:
        find_y = False
        for port in ports:
            if port == start_port:
                find_y = True
                break
        if find_y:
            start_port = start_port + 1
        else:
            return start_port


def mount_controller_routes(app: FastAPI,
                            cfg: Dynaconf,
                            ):
    from fastchat.serve.controller import logger
    from hpdeploy.llm_model.shutdown_serve import shutdown_worker_serve, shutdown_serve
    from hpdeploy.llm_model.launch_all_serve import launch_worker, launch_api_server

    global global_worker_dict

    def check_worker_start_status(model):
        def action():
            nonlocal model
            print(f"-----------------------check status of: {model} -----------------------")
            worker_address = app._controller.get_worker_address(model)
            if not worker_address:
                print(f"-------------------model start failed: {model} --------------------")
                stop_model(model)

        threading.Timer(60, action).start()  # 延时x秒后执行action函数

    def run_default_models():
        default_run = cfg.get("llm.default_run", [])
        res = {}
        if default_run:
            for m in default_run:
                res[m] = start_model(m)
        return BaseResponse(data=res)

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

    def list_online_embed_models() -> BaseResponse:
        """
        从本地获取configs中配置的online embedding模型列表
        """
        from hpdeploy.llm_model import model_workers
        ret = {}
        for k, v in cfg.get("llm.online_model_cfg", {}).items():
            if provider := v.get("provider"):
                worker_class = getattr(model_workers, provider, None)
                if worker_class is not None and worker_class.can_embedding():
                    ret[k] = provider
        return BaseResponse(data=ret)

    def start_model(
            model_name: str = Body(None, description="启动该模型"),
            placeholder: str = Body(None, description="占位用，无实际效果"),
    ) -> Dict:
        # return model_worker_ctl(["start_worker", model_name])
        ret = model_name.split("@")
        port = None
        if len(ret) == 2:
            model_name, port = ret
        else:
            port = find_use_port()
        ret, msg = launch_worker(f"{model_name}@{port}")
        global_worker_dict[model_name] = {"port": port, "success": ret}
        if ret:
            return {"success": True, "code": 200, "msg": "success"}
        else:
            check_worker_start_status(model_name)
            return {"success": False, "code": 501, "msg": msg}

    def stop_model(
            model_name: str = Body(None, description="停止该模型"),
            placeholder: str = Body(None, description="占位用，无实际效果"),
    ) -> Dict:
        # return model_worker_ctl(["stop_worker", model_name])
        if model_name in global_worker_dict:
            del global_worker_dict[model_name]
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

    def api_server_ctrl(
            ctrl: str = Body("start", description="停止该模型"),
            port: Optional[int] = Body(None, description="端口号"),
    ) -> Dict:
        if ctrl == "start":
            server_str_args = ""
            if port is not None and port >= 3000:
                server_str_args = f" --port {port}"
            launch_api_server(server_str_args)
        elif ctrl == "stop":
            shutdown_serve("api")
        return {"success": False, "code": 500, "msg": f"unknown ctrl: {ctrl}"}

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

    app.post("/run_default_models",
             tags=["LLM Management"],
             response_model=BaseResponse,
             summary="启动配置的所有llm模型"
             )(run_default_models)

    app.post("/api_server_ctrl",
             tags=["Agent Management"],
             response_model=BaseResponse,
             summary="启停api server"
             )(api_server_ctrl)

    def init_get_registed_workers_info():
        global global_worker_dict
        # available_models = app._controller.list_models()
        # worker_address = app._controller.get_worker_address(available_models[0])
        print("--------------------init_get_registed_workers_info--------------------------")
        print(app._controller.worker_info)
        for w_name, w_info in app._controller.worker_info.items():
            port = w_name.split(":")[2]
            for m in w_info.model_names:
                global_worker_dict[m] = {"port": port, "success": True}
        print("--------------------global_worker_dict--------------------------")
        print(global_worker_dict)

    t = threading.Timer(15, init_get_registed_workers_info)  # 延时x秒后执行action函数
    t.start()
    threading.Timer(25, init_get_registed_workers_info).start()
    threading.Timer(35, init_get_registed_workers_info).start()
    threading.Timer(45, init_get_registed_workers_info).start()

    app.start_model = start_model
    app.stop_model = stop_model
    app.replace_model = replace_model
    return app
