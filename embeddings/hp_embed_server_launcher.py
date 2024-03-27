import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
runtime_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__current_script_path)))
sys.path.append(runtime_root_dir)

from dynaconf import Dynaconf


def mount_more_routes(app):
    from hpdeploy.embeddings.embeddings_api import aembed_texts
    from fuxi.utils.api_base import (BaseResponse, ListResponse)
    from hpdeploy.embeddings.config import get_config_embed_models

    def list_embed_models() -> BaseResponse:
        """
        从本地获取configs中配置的embedding模型列表
        """
        return BaseResponse(data=get_config_embed_models())

    app.post("/embed/encode",
             tags=["Embed"],
             summary="call embeddings to encode texts, return List[List[float]]"
             )(aembed_texts)

    app.post("/list_embed_models",
             tags=["Embed"],
             response_model=BaseResponse,
             summary="查看配置的所有embeddings模型"
             )(list_embed_models)

    # app.post("/list_online_embed_models",
    #          tags=["Embed"],
    #          response_model=BaseResponse,
    #          summary="查看配置的所有online embeddings模型"
    #          )(list_online_embed_models)


def init_before_mount_routes(cfg: Dynaconf, app):
    from hpdeploy.embeddings.embeddings_api import load_local_embeddings
    default_run = cfg.get("embed.default_run", [])
    for embed in default_run:
        load_local_embeddings(embed)


def base_init_1(cfg: Dynaconf):
    from fuxi.utils.fastapi_tool import MakeFastAPIOffline, create_app

    cross_domain = cfg.get("embed.server.cross_domain", cfg.get("root.cross_domain", True))
    version = cfg.get("embed.server.version", cfg.get("root.version", "1.0.0"))
    app = create_app([], version=version, title="风后AI-Embeddings API Server", cross_domain=cross_domain)

    MakeFastAPIOffline(app)

    return app


def base_init_0(cfg: Dynaconf, log_level):
    app = base_init_1(cfg)
    init_before_mount_routes(cfg, app)
    mount_more_routes(app)

    host = cfg.get("embed.server.host")
    port = cfg.get("embed.server.port")
    if host == "localhost" or host == "127.0.0.1":
        host = "0.0.0.0"

    from fuxi.utils.fastapi_tool import run_api
    run_api(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=cfg.get("embed.server.ssl_keyfile", cfg.get("root.ssl_keyfile", None)),
        ssl_certfile=cfg.get("embed.server.ssl_certfile", cfg.get("root.ssl_certfile", None)),
    )


def init_api_server():
    import argparse
    from fuxi.utils.runtime_conf import get_runtime_root_dir
    from fuxi.utils.torch_helper import detect_device

    print(get_runtime_root_dir())
    cfg = Dynaconf(
        envvar_prefix="JIAN",
        root_path=get_runtime_root_dir(),
        settings_files=['conf/llm_model.yml', 'conf/settings.yaml'],
    )
    import hpdeploy.embeddings.config as bc
    bc.cfg = cfg

    log_level = cfg.get("embed.server.log_level", "info")
    verbose = True if log_level == "debug" else False
    host = cfg.get("embed.server.host", "0.0.0.0")
    port = cfg.get("embed.server.port", 21199)

    parser = argparse.ArgumentParser(prog='FenghouAI',
                                     description='About FenghouAI Embeddings API')
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument(
        "-v",
        "--verbose",
        help="增加log信息",
        dest="verbose",
        type=bool,
        default=verbose,
    )
    # 初始化消息
    args = parser.parse_args()
    host = args.host
    port = args.port
    if args.verbose:
        log_level = "debug"
        cfg["embed.server.log_level"] = "debug"

    cfg["embed.server.host"] = host
    cfg["embed.server.port"] = port

    cuda = args.cuda
    if cuda >= 0:
        cfg["embed.device"] = f"cuda:{cuda}"
    else:
        device = cfg.get("embed.device", None)
        if device is None or device == "auto":
            cfg["embed.device"] = detect_device()

    base_init_0(cfg, log_level)


if __name__ == "__main__":
    init_api_server()
