import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

import subprocess
from common.utils import DEFAULT_LOG_PATH

LOGDIR = DEFAULT_LOG_PATH

# 0,controller, model_worker, openai_api_server
# 1, cmd options
# 2,LOGDIR
# 3, log file name
base_launch_sh = "nohup python3 {0} {1} >{2}/{3}.log 2>&1 &"

# 0 LOGDIR
# ! 1 log file name
# 2 controller, worker, openai_api_server
base_check_sh = """i=0;
                while [ `grep -c "Uvicorn running on" {1}/{2}.log` -eq '0' ];do
                        if [ $i -gt {0} ] 
                        then 
                            echo "wait timeout({0})!";
                            exit 2;
                        fi
                        sleep 1s;
                        echo "wait {3} running";
                        i=`expr $i + 1`;
                done
                echo '{3} running' """

base_check_model_sh = """i=0;
                while [ `grep -c "Uvicorn running on" {1}/{2}.log` -eq '0' ];do
                        if `ps -ef |grep fschat_worker_launcher.py|grep "{4}"| grep -v grep > /dev/null`
                        then c=0;
                        else 
                            echo "process {3}-{4} is exited!";
                            exit 1;
                        fi
                        if [ $i -gt {0} ] 
                        then 
                            echo "wait timeout({0})!";
                            exit 2;
                        fi
                        sleep 2s;
                        echo "wait {3}-{4} running";
                        i=`expr $i + 1`;
                done
                echo '{3}-{4} running' """


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


from llm_model.shutdown_serve import check_worker_processes
def launch_worker(model, worker_str_args: str = "", wait_times: int = 60):
    if check_worker_processes(model):
        msg = f"Skip, the {model} worker_processes is already existed!"
        print(msg)
        return False, msg
    log_name = model
    # args.model_path, args.worker_host, args.worker_port = item.split("@")
    print("*" * 80)
    worker_str_args = f" --model {model} {worker_str_args}"
    print(worker_str_args)
    # "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"
    worker_sh = base_launch_sh.format(
        "llm_model/fschat_worker_launcher.py", worker_str_args, LOGDIR, f"worker_{log_name}"
    )
    worker_check_sh = base_check_model_sh.format(int(wait_times / 2), LOGDIR, f"worker_{log_name}", "model_worker", model)
    print(f"executing worker_sh: {worker_sh}")
    subprocess.run(worker_sh, shell=True, check=True)
    try:
        subprocess.run(worker_check_sh, shell=True, check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        print(e)
        return False, f"start {model} worker fail, see the log for details"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

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
        type=bool,
        default=False,
    )
    model_worker_args = ["wp", "verbose"]

    # ---------------------------------------------MAIN---------------------------------------------------
    args = parser.parse_args()


    def launch_all():
        controller_str_args = string_args(args, controller_args)
        # "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"
        controller_sh = base_launch_sh.format(
            "llm_model/fschat_controller_launcher.py", controller_str_args, LOGDIR, "controller"
        )
        controller_check_sh = base_check_sh.format(10, LOGDIR, "controller", "controller")
        print(f"executing controller_sh: {controller_sh}")
        # print(f"watch controller log: {controller_check_sh}")
        subprocess.run(controller_sh, shell=True, check=True)
        subprocess.run(controller_check_sh, shell=True, check=True)

        if args.model:
            worker_str_args = string_args(args, model_worker_args)
            if isinstance(args.model, str):
                launch_worker(args.model, worker_str_args)
            else:
                for idx, item in enumerate(args.model):
                    print(f"loading {idx}th model:{item}")
                    launch_worker(item, worker_str_args)

        server_str_args = string_args(args, openaiapi_server_args)
        server_sh = base_launch_sh.format(
            "llm_model/fschat_openai_api_server_launcher.py", server_str_args, LOGDIR, "openai_api_server"
        )
        server_check_sh = base_check_sh.format(10, LOGDIR, "openai_api_server", "openai_api_server")
        subprocess.run(server_sh, shell=True, check=True)
        subprocess.run(server_check_sh, shell=True, check=True)


    launch_all()
