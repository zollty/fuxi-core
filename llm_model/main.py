import asyncio
import multiprocessing as mp
import os
import sys
from multiprocessing import Process
from datetime import datetime

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)
print(RUNTIME_ROOT_DIR)

import argparse
from typing import List, Dict
from fastapi import FastAPI
from common.utils import logger, LOG_VERBOSE, DEFAULT_LOG_PATH
import fastchat.constants
from dynaconf import Dynaconf

fastchat.constants.LOGDIR = DEFAULT_LOG_PATH


def parse_args() -> argparse.ArgumentParser:
    from common.model_args import add_model_args
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="Print debugging messages"
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="增加log信息",
        dest="verbose",
    )
    parser.add_argument(
        "-fs",
        "--fastchat",
        action="store_true",
        help="run fastchat's controller/openai_api servers",
        dest="fastchat",
    )
    parser.add_argument(
        "-w",
        "--worker",
        action="store_true",
        help="run fastchat's worker servers",
        dest="worker",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    return args, parser


def dump_server_info(cfg: Dynaconf, after_start=False):
    from common.utils import RUNTIME_ROOT_DIR
    import platform
    import langchain
    import fastchat
    from common.utils import detect_device

    server_info = []

    server_info.append("\n")
    server_info.append("=" * 30 + "Chatchat Configuration" + "=" * 30)
    server_info.append(f"操作系统：{platform.platform()}.")
    server_info.append(f"python版本：{sys.version}")
    server_info.append(f"项目版本：{cfg.root.version}")
    server_info.append(f"langchain版本：{langchain.__version__}. fastchat版本：{fastchat.__version__}")
    server_info.append("\n")

    if models := cfg.get("root.default_start_model"):
        server_info.append(f"当前启动的LLM模型：{models} @ {detect_device()}")

    print(''.join(server_info))
    if after_start:
        with open(RUNTIME_ROOT_DIR + '/logs/start_info.txt', 'r') as f:
            print(f.read())
    print("=" * 30 + "FenghouAI Configuration" + "=" * 30)
    print("\n")


async def start_main_server():
    import time
    import signal
    from llm_model.fschat_controller_launcher import run_controller
    from llm_model.fschat_openai_api_server_launcher import run_openai_api_server
    from llm_model.fschat_worker_launcher import run_model_worker
    from common.utils import RUNTIME_ROOT_DIR
    from dynaconf import Dynaconf

    cfg = Dynaconf(
        envvar_prefix="FUXI",
        root_path=RUNTIME_ROOT_DIR,
        settings_files=['llm_model/conf_llm_model.yml', 'settings.yaml'],
    )

    with open(RUNTIME_ROOT_DIR + '/logs/start_info.txt', "w") as f:
        f.truncate(0)

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()

    queue = manager.Queue()
    args, parser = parse_args()

    dump_server_info(cfg)
    if len(sys.argv) > 1:
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {DEFAULT_LOG_PATH}")

    processes = {"online_api": {}, "model_worker": {}}

    def process_count():
        return len(processes) + len(processes["online_api"]) + len(processes["model_worker"]) - 2

    if args.verbose or LOG_VERBOSE:
        log_level = "INFO"
    else:
        log_level = "ERROR"

    controller_started = manager.Event()
    if args.fastchat:
        process = Process(
            target=run_controller,
            name=f"controller",
            kwargs=dict(
                manager_queue=queue,
                started_event=controller_started),
            daemon=True,
        )
        processes["controller"] = process

        process = Process(
            target=run_openai_api_server,
            name=f"openai_api",
            daemon=True,
        )
        processes["openai_api"] = process

        # process = Process(
        #     target=run_openai_api,
        #     name=f"openai_api",
        #     daemon=True,
        # )
        # processes["openai_api"] = process
    model_worker_started = []
    if dm := cfg.get("root.default_start_model"):
        for new_model_name in dm:
            e = manager.Event()
            model_worker_started.append(e)
            process = Process(
                target=run_model_worker,
                name=f"model_worker - {new_model_name}",
                kwargs=dict(model_name=new_model_name,
                            started_event=e),
                daemon=True,
            )
            processes["model_worker"][new_model_name] = process

    if process_count() == 0:
        parser.print_help()
    else:
        try:
            # 保证任务收到SIGINT后，能够正常退出
            if p := processes.get("controller"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                controller_started.wait()  # 等待controller启动完成

            if p := processes.get("openai_api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("model_worker", {}).items():
                logger.info(f"准备启动新模型进程：{p.name}")
                p.start()
                p.name = f"{p.name} ({p.pid})"

            # 等待所有model_worker启动完成
            for e in model_worker_started:
                e.wait()
            # for process in processes.get("model_worker", {}).values():
            #     process.join()
            # for process in processes.get("online_api", {}).values():
            #     process.join()

            # for name, process in processes.items():
            #     if name not in ["model_worker", "online_api"]:
            #         if isinstance(p, dict):
            #             for work_process in p.values():
            #                 work_process.join()
            #         else:
            #             process.join()

            dump_server_info(cfg, after_start=True)
            while True:
                msg = queue.get()  # 收到切换模型的消息
                e = manager.Event()
                logger.info("收到消息", msg)
                #  managerQueue.put([model_name, "start", new_model_name])
                if isinstance(msg, list):
                    cmd = msg[0]
                    if cmd == "start_worker":  # 运行新模型
                        new_model_name = msg[1]
                        logger.info(f"准备启动新模型进程：{new_model_name}")
                        start_time = datetime.now()
                        process = Process(
                            target=run_model_worker,
                            name=f"model_worker - {new_model_name}",
                            kwargs=dict(model_name=new_model_name,
                                        started_event=e),
                            daemon=True,
                        )
                        process.start()
                        process.name = f"{process.name} ({process.pid})"
                        processes["model_worker"][new_model_name] = process
                        e.wait()
                        timing = datetime.now() - start_time
                        logger.info(f"成功启动新模型进程：{new_model_name}。用时：{timing}。")
                    elif cmd == "stop":
                        model_name = msg[1]
                        if process := processes["model_worker"].get(model_name):
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            logger.info(f"停止模型进程：{model_name}")
                        else:
                            logger.error(f"未找到模型进程：{model_name}")
                    elif cmd == "replace":
                        model_name = msg[1]
                        new_model_name = msg[2]
                        if process := processes["model_worker"].pop(model_name, None):
                            logger.info(f"停止模型进程：{model_name}")
                            start_time = datetime.now()
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            process = Process(
                                target=run_model_worker,
                                name=f"model_worker - {new_model_name}",
                                kwargs=dict(model_name=new_model_name,
                                            started_event=e),
                                daemon=True,
                            )
                            process.start()
                            process.name = f"{process.name} ({process.pid})"
                            processes["model_worker"][new_model_name] = process
                            e.wait()
                            timing = datetime.now() - start_time
                            logger.info(f"成功启动新模型进程：{new_model_name}。用时：{timing}。")
                        else:
                            logger.error(f"未找到模型进程：{model_name}")


        except Exception as e:
            logger.error(e)
            logger.warning("Caught KeyboardInterrupt! Setting stop event...")
        finally:
            # Send SIGINT if process doesn't exit quickly enough, and kill it as last resort
            # .is_alive() also implicitly joins the process (good practice in linux)
            # while alive_procs := [p for p in processes.values() if p.is_alive()]:

            for p in processes.values():
                logger.warning("Sending SIGKILL to %s", p)
                # Queues and other inter-process communication primitives can break when
                # process is killed, but we don't care here

                if isinstance(p, dict):
                    for process in p.values():
                        process.kill()
                else:
                    p.kill()

            for p in processes.values():
                logger.info("Process status: %s", p)


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
    # 同步调用协程代码
    loop.run_until_complete(start_main_server())
