import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

from typing import Any, List, Optional, Dict
import multiprocessing as mp
from fastapi import FastAPI, Body
from common.fastapi_tool import set_app_event
from common.api_base import (BaseResponse, ListResponse)
from dynaconf import Dynaconf


def mount_controller_routes(app: FastAPI,
                            manager_queue: mp.Queue = None,
                            ):
    from fastchat.serve.controller import logger
    def model_worker_ctl(
            msg: List[str] = Body(..., description="参数"),
    ) -> Dict:
        # 与manager进程通信
        manager_queue.put(msg)
        logger.warn(f"execute: {msg}")
        return {"code": 200, "msg": "done"}

    def start_model(
            model_name: str = Body(None, description="启动该模型"),
    ) -> Dict:
        return model_worker_ctl(["start_worker", model_name])

    def stop_model(
            model_name: str = Body(None, description="停止该模型"),
    ) -> Dict:
        return model_worker_ctl(["stop_worker", model_name])

    def replace_model(
            model_name: str = Body(None, description="停止该模型"),
            new_model_name: str = Body(None, description="替换该模型"),
    ) -> Dict:
        return model_worker_ctl(["replace_worker", model_name, new_model_name])

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


def create_controller_app(cfg: Dynaconf, log_level, manager_queue: mp.Queue = None):
    from common.utils import DEFAULT_LOG_PATH
    from common.fastapi_tool import set_httpx_config, MakeFastAPIOffline
    import sys
    import fastchat.constants
    from fastchat.serve.controller import app, Controller, logger
    from fastapi.middleware.cors import CORSMiddleware

    fastchat.constants.LOGDIR = DEFAULT_LOG_PATH
    logger.setLevel(log_level.upper())

    dispatch_method = cfg.get("llm.controller.dispatch_method", "shortest_queue")
    cross_domain = cfg.get("llm.controller.cross_domain", cfg.get("root.cross_domain", True))

    controller = Controller(dispatch_method)

    app.title = "FastChat Controller"
    app.version = fastchat.__version__

    mount_controller_routes(app, manager_queue)

    if cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    set_httpx_config()

    sys.modules["fastchat.serve.controller"].controller = controller
    app._controller = controller
    MakeFastAPIOffline(app)

    return app


def run_controller(manager_queue: mp.Queue = None, started_event: mp.Event = None):
    from common.utils import RUNTIME_ROOT_DIR
    from common.fastapi_tool import run_api

    print(RUNTIME_ROOT_DIR)
    cfg = Dynaconf(
        envvar_prefix="FUXI",
        root_path=RUNTIME_ROOT_DIR,
        settings_files=['llm_model/conf_llm_model.yml', 'settings.yaml'],
    )

    log_level = cfg.get("llm.controller.log_level", cfg.get("root.log_level", "INFO")).upper()
    host = cfg.get("llm.controller.host", "0.0.0.0")
    port = cfg.get("llm.controller.port", 21001)

    app = create_controller_app(cfg, log_level, manager_queue)

    set_app_event(app, started_event)

    with open(RUNTIME_ROOT_DIR + '/logs/start_info.txt', 'a') as f:
        f.write(f"    FenghouAI Controller Server (fastchat): http://{host}:{port}\n")

    run_api(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=cfg.get("controller.ssl_keyfile"),
        ssl_certfile=cfg.get("controller.ssl_certfile"),
    )


if __name__ == "__main__":
    run_controller()
