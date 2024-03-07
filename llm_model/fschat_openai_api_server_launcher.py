import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

from dynaconf import Dynaconf


def create_openai_api_server_app(cfg: Dynaconf, log_level):
    from fastapi.middleware.cors import CORSMiddleware
    from common.utils import DEFAULT_LOG_PATH, OPEN_CROSS_DOMAIN
    from common.fastapi_tool import set_httpx_config, MakeFastAPIOffline
    import sys
    import fastchat.constants
    fastchat.constants.LOGDIR = DEFAULT_LOG_PATH
    from fastchat.serve.openai_api_server import app, app_settings
    from fastchat.utils import build_logger
    logger = build_logger("openai_api", "openai_api.log")
    sys.modules["fastchat.serve.openai_api_server"].logger = logger

    logger.setLevel(log_level.upper())

    controller_address = cfg.get("llm.openai_api_server.controller_addr")
    cross_domain = cfg.get("llm.openai_api_server.cross_domain", cfg.get("root.cross_domain", True))

    app_settings.controller_address = controller_address
    app_settings.api_keys = cfg.get("llm.openai_api_server.api_keys", "")

    app.title = "FastChat OpeanAI API Server"
    app.version = fastchat.__version__

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


def run_openai_api_server():
    from common.utils import RUNTIME_ROOT_DIR
    from common.fastapi_tool import run_api

    print(RUNTIME_ROOT_DIR)
    cfg = Dynaconf(
        envvar_prefix="FUXI",
        root_path=RUNTIME_ROOT_DIR,
        settings_files=['llm_model/conf_llm_model.yml', 'settings.yaml'],
    )

    log_level = cfg.get("llm.openai_api_server.log_level", "info")
    host = cfg.get("llm.openai_api_server.host", "0.0.0.0")
    port = cfg.get("llm.openai_api_server.port", 8000)

    app = create_openai_api_server_app(cfg, log_level)

    with open(RUNTIME_ROOT_DIR + '/logs/start_info.txt', 'a') as f:
        f.write(f"    FenghouAI OpeanAI API Server (fastchat): http://{host}:{port}")
    run_api(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=cfg.get("llm.openai_api_server.ssl_keyfile"),
        ssl_certfile=cfg.get("llm.openai_api_server.ssl_certfile"),
    )


if __name__ == "__main__":
    run_openai_api_server()
