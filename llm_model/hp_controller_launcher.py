import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
runtime_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__current_script_path)))
sys.path.append(runtime_root_dir)

from dynaconf import Dynaconf


def create_controller_app(cfg: Dynaconf, log_level):
    from fuxi.utils.runtime_conf import get_default_log_path
    from fuxi.utils.fastapi_tool import MakeFastAPIOffline
    import sys
    import fastchat.constants
    from fastchat.serve.controller import app, Controller, logger
    from fastapi.middleware.cors import CORSMiddleware
    from hpdeploy.llm_model.controller import mount_controller_routes

    fastchat.constants.LOGDIR = get_default_log_path()
    logger.setLevel(log_level.upper())

    dispatch_method = cfg.get("llm.controller.dispatch_method", "shortest_queue")
    cross_domain = cfg.get("llm.controller.cross_domain", cfg.get("root.cross_domain", True))

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller
    app._controller = controller

    app.title = "伏羲AI Controller Server"
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

    MakeFastAPIOffline(app)

    return app


def run_controller():
    from fuxi.utils.runtime_conf import get_runtime_root_dir
    from fuxi.utils.fastapi_tool import run_api
    import threading

    import argparse

    print(get_runtime_root_dir())
    cfg = Dynaconf(
        envvar_prefix="HP",
        root_path=get_runtime_root_dir(),
        settings_files=['conf/llm_model.yml', 'settings.yaml'],  # 后者优先级高，以一级key覆盖前者（一级key相同的，前者不生效）
    )

    log_level = cfg.get("llm.controller.log_level", "INFO").upper()
    host = cfg.get("llm.controller.host", "0.0.0.0")
    port = cfg.get("llm.controller.port", 21001)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd",
        "--run-default",
        help="运行配置的默认模型",
        dest="run_default",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="增加log信息",
        dest="verbose",
        type=bool,
        default=False,
    )
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)

    # 初始化消息
    args = parser.parse_args()
    host = args.host
    port = args.port

    cfg["llm.controller.host"] = host
    cfg["llm.controller.port"] = port

    from fuxi.utils.fastapi_tool import set_httpx_config
    set_httpx_config()

    app = create_controller_app(cfg, log_level)

    default_run = cfg.get("llm.default_run", [])
    if default_run and args.run_default:
        def my_function():
            for m in default_run:
                app.start_model(m)

        my_thread = threading.Thread(target=my_function)
        my_thread.start()

    with open(get_runtime_root_dir() + '/logs/start_info.txt', 'a') as f:
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
