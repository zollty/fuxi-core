import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

from common.conf import Cfg


def create_controller_app(cfg: Cfg):
    from common.utils import DEFAULT_LOG_PATH, VERSION, OPEN_CROSS_DOMAIN
    from common.fastapi_tool import set_httpx_config, MakeFastAPIOffline
    import sys
    import fastchat.constants
    from fastchat.serve.controller import app, Controller, logger
    from fastapi.middleware.cors import CORSMiddleware

    fastchat.constants.LOGDIR = DEFAULT_LOG_PATH
    log_level = cfg.get("controller.log_level", "info")
    logger.setLevel(log_level.upper())

    dispatch_method = cfg.get("controller.dispatch_method", "shortest_queue")
    cross_domain = cfg.get("controller.cross_domain", OPEN_CROSS_DOMAIN)

    controller = Controller(dispatch_method)

    app.title = "FastChat Controller"
    app.version = VERSION

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


if __name__ == "__main__":
    from common.utils import RUNTIME_ROOT_DIR
    from common.fastapi_tool import run_api

    print(RUNTIME_ROOT_DIR)
    cfg = Cfg(RUNTIME_ROOT_DIR + "/conf_llm_model.toml")

    log_level = cfg.get("controller.log_level", "info")
    host = cfg.get("controller.host", "0.0.0.0")
    port = cfg.get("controller.port", 21001)

    app = create_controller_app(cfg)

    run_api(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=cfg.get("controller.ssl_keyfile"),
        ssl_certfile=cfg.get("controller.ssl_certfile"),
    )