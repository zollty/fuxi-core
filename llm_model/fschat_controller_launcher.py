import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

from typing import Any, List, Optional, Dict
from dynaconf import Dynaconf


def create_controller_app(cfg: Dynaconf, log_level):
    from common.utils import DEFAULT_LOG_PATH
    from common.fastapi_tool import set_httpx_config, MakeFastAPIOffline
    import sys
    import fastchat.constants
    from fastchat.serve.controller import app, Controller, logger
    from fastapi.middleware.cors import CORSMiddleware
    from llm_model.controller import mount_controller_routes

    fastchat.constants.LOGDIR = DEFAULT_LOG_PATH
    logger.setLevel(log_level.upper())

    dispatch_method = cfg.get("llm.controller.dispatch_method", "shortest_queue")
    cross_domain = cfg.get("llm.controller.cross_domain", cfg.get("root.cross_domain", True))

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller
    app._controller = controller

    app.title = "FastChat Controller"
    app.version = fastchat.__version__

    mount_controller_routes(app, cfg)

    if cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    set_httpx_config()

    MakeFastAPIOffline(app)

    return app


def run_controller():
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

    app = create_controller_app(cfg, log_level)

    default_run = cfg.get("llm.default_run", [])
    if default_run:
        import threading
        def my_function():
            for m in default_run:
                app.start_model(m)

        my_thread = threading.Thread(target=my_function)
        my_thread.start()

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
