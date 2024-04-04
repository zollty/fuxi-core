import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
runtime_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__current_script_path)))
sys.path.append(runtime_root_dir)

from dynaconf import Dynaconf


def mount_more_routes(app):
    from hpdeploy.rerank.reranker_api import apredict
    from fuxi.utils.api_base import (BaseResponse, ListResponse)
    from hpdeploy.rerank.config import get_config_rerank_models

    def list_rerank_models() -> BaseResponse:
        """
        从本地获取configs中配置的embedding模型列表
        """
        return BaseResponse(data=get_config_rerank_models())

    app.post("/rerank/predict",
             tags=["Rerank"],
             summary="对文本进行重排序检索。返回数据格式：List[float]"
             )(apredict)

    app.post("/list_rerank_models",
             tags=["Rerank"],
             response_model=BaseResponse,
             summary="查看配置的所有reranker模型"
             )(list_rerank_models)


def init_before_mount_routes(cfg: Dynaconf, app):
    from hpdeploy.rerank.reranker_api import load_local_reranker
    default_run = cfg.get("rerank.default_run", [])
    for reranker in default_run:
        load_local_reranker(reranker)


def base_init_1(cfg: Dynaconf):
    from fuxi.utils.fastapi_tool import MakeFastAPIOffline, create_app

    cross_domain = cfg.get("rerank.server.cross_domain", cfg.get("root.cross_domain", True))
    version = cfg.get("rerank.server.version", cfg.get("root.version", "1.0.0"))
    app = create_app([], version=version, title="风后AI-Reranker API Server", cross_domain=cross_domain)

    MakeFastAPIOffline(app)

    return app


def base_init_0(cfg: Dynaconf, log_level):
    app = base_init_1(cfg)
    init_before_mount_routes(cfg, app)
    mount_more_routes(app)

    host = cfg.get("rerank.server.host")
    port = cfg.get("rerank.server.port")
    if host == "localhost" or host == "127.0.0.1":
        host = "0.0.0.0"

    from fuxi.utils.fastapi_tool import run_api
    run_api(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=cfg.get("rerank.server.ssl_keyfile", cfg.get("root.ssl_keyfile", None)),
        ssl_certfile=cfg.get("rerank.server.ssl_certfile", cfg.get("root.ssl_certfile", None)),
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
    import hpdeploy.rerank.config as bc
    bc.cfg = cfg

    log_level = cfg.get("rerank.server.log_level", "info")
    verbose = True if log_level == "debug" else False
    host = cfg.get("rerank.server.host", "0.0.0.0")
    port = cfg.get("rerank.server.port", 21199)

    parser = argparse.ArgumentParser(prog='FenghouAI',
                                     description='About FenghouAI Reranker API')
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
        cfg["rerank.server.log_level"] = "debug"

    cfg["rerank.server.host"] = host
    cfg["rerank.server.port"] = port

    cuda = args.cuda
    if cuda >= 0:
        cfg["rerank.device"] = f"cuda:{cuda}"
    else:
        device = cfg.get("rerank.device", None)
        if device is None or device == "auto":
            cfg["rerank.device"] = detect_device()

    device = cfg["rerank.device"]
    print(f"-----------------------------use device: {device}")

    base_init_0(cfg, log_level)


if __name__ == "__main__":
    init_api_server()
