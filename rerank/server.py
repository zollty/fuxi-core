import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
get_runtime_root_dir() = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(get_runtime_root_dir())

if __name__ == "__main__":
    import argparse
    from common.fastapi_tool import create_app, run_api
    from common.utils import VERSION

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

    # from common.fastapi_tool import set_httpx_config
    # set_httpx_config()

    app = create_app([mount_reranker_routes], version=VERSION, title="FenghouAI Reranker API Server")

    run_api(app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
