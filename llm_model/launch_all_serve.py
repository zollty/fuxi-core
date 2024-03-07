import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

import subprocess
import argparse
from common.utils import DEFAULT_LOG_PATH

LOGDIR = DEFAULT_LOG_PATH

parser = argparse.ArgumentParser()

# 0,controller, model_worker, openai_api_server
# 1, cmd options
# 2,LOGDIR
# 3, log file name
base_launch_sh = "nohup python3 -m {0} {1} >{2}/{3}.log 2>&1 &"

# 0 LOGDIR
# ! 1 log file name
# 2 controller, worker, openai_api_server
base_check_sh = """while [ `grep -c "Uvicorn running on" {0}/{1}.log` -eq '0' ];do
                        sleep 1s;
                        echo "wait {2} running"
                done
                echo '{2} running' """


def string_args(args, args_list):
    args_str = ""
    for key, value in args._get_kwargs():
        key = key.replace("_", "-")
        if key not in args_list:
            continue

        # key = key.split("-")[-1] if re.search("port|host", key) else key
        if not value:
            pass
        if key == "op" or key == "wp" or key == "cp":
            key = "custom-config"
        # 1==True ->  True
        elif isinstance(value, bool) and value == True:
            args_str += f" --{key} "
        elif (
                isinstance(value, list)
                or isinstance(value, tuple)
                or isinstance(value, set)
        ):
            value = " ".join(value)
            args_str += f" --{key} {value} "
        else:
            args_str += f" --{key} {value} "

    return args_str


parser.add_argument(
    "-cp",
    "--controller-config-path",
    help="custom controller config path",
    dest="cp",
    default=None,
)
controller_args = ["cp", "verbose"]

parser.add_argument(
    "-op",
    "--openaiapi-config-path",
    help="custom openai_api config path",
    dest="op",
    default=None,
)
openaiapi_server_args = ["op", "verbose"]

parser.add_argument(
    "-wp",
    "--worker-config-path",
    help="custom model worker config path",
    dest="wp",
    default=None,
)
parser.add_argument(
    "-m",
    "--model",
    nargs="+",
    type=str,
    help="custom default start models",
    dest="model",
    default=None,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="增加log信息",
    dest="verbose",
    default=None,
)
model_worker_args = ["wp", "verbose"]

# ---------------------------------------------MAIN---------------------------------------------------
args = parser.parse_args()


def launch_worker(model):
    log_name = model
    # args.model_path, args.worker_host, args.worker_port = item.split("@")
    print("*" * 80)
    worker_str_args = f" --model {model} " + string_args(args, model_worker_args)
    print(worker_str_args)
    # "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"
    worker_sh = base_launch_sh.format(
        "llm_model/fschat_worker_launcher.py", worker_str_args, LOGDIR, f"worker_{log_name}"
    )
    worker_check_sh = base_check_sh.format(LOGDIR, f"worker_{log_name}", "model_worker")
    print(f"executing worker_sh: {worker_sh}")
    subprocess.run(worker_sh, shell=True, check=True)
    subprocess.run(worker_check_sh, shell=True, check=True)


def launch_all():
    LOGDIR = "./logs/"

    controller_str_args = string_args(args, controller_args)
    # "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"
    controller_sh = base_launch_sh.format(
        "llm_model/fschat_controller_launcher.py", controller_str_args, LOGDIR, "controller"
    )
    controller_check_sh = base_check_sh.format(LOGDIR, "controller", "controller")
    print(f"executing controller_sh: {controller_sh}")
    print(f"watch controller log: {controller_check_sh}")
    subprocess.run(controller_sh, shell=True, check=True)
    subprocess.run(controller_check_sh, shell=True, check=True)

    if args.model:
        if isinstance(args.model, str):
            launch_worker(args.model)
        else:
            for idx, item in enumerate(args.model):
                print(f"loading {idx}th model:{item}")
                launch_worker(item)

    server_str_args = string_args(args, openaiapi_server_args)
    server_sh = base_launch_sh.format(
        "llm_model/fschat_openai_api_server_launcher.py", server_str_args, LOGDIR, "openai_api_server"
    )
    server_check_sh = base_check_sh.format(
        LOGDIR, "openai_api_server", "openai_api_server"
    )
    subprocess.run(server_sh, shell=True, check=True)
    subprocess.run(server_check_sh, shell=True, check=True)


if __name__ == "__main__":
    launch_all()
