import argparse
from common.fastapi_tool import create_app, run_api

VERSION = "1.0.0"
# API 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Fenghou-AI',
                                     description='About FenghouAI, local knowledge QA'
                                                 ' ｜ 基于本地知识库的问答')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7765)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    from rerank.api import mount_reranker_routes
    from common.fastapi_tool import set_httpx_config

    set_httpx_config()

    app = create_app([mount_reranker_routes])

    run_api(app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
