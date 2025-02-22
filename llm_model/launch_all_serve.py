import sys
import os
import time

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
runtime_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__current_script_path)))
sys.path.append(runtime_root_dir)

import subprocess
from typing import Optional
from fuxi.utils.runtime_conf import get_default_log_path

LOGDIR = get_default_log_path()

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
                        if `ps -ef |grep hp_worker_launcher.py|grep "{4}"| grep -v grep > /dev/null`
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
        if isinstance(value, bool) and value == True:
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


from hpdeploy.llm_model.shutdown_serve import check_worker_processes


def get_gpus(model_name):
    from fuxi.utils.runtime_conf import get_runtime_root_dir
    from dynaconf import Dynaconf

    cfg = Dynaconf(
        envvar_prefix="HP",
        root_path=get_runtime_root_dir(),
        settings_files=['conf/llm_model.yml', 'conf/settings.yaml'],  # 后者优先级高，以一级key覆盖前者（一级key相同的，前者不生效）
    )
    gpus = cfg.get("llm.model_cfg")[model_name]["base"]["gpus"]
    return f"CUDA_VISIBLE_DEVICES={gpus} "

def launch_worker(model, worker_str_args: str = "", wait_times: int = 60):
    ret = model.split("@")
    model_name = model
    port = None
    if len(ret) == 2:
        model_name, port = ret
    if check_worker_processes(model_name):
        msg = f"Skip, the {model_name} worker_processes is already existed!"
        print(msg)
        return False, msg
    log_name = model_name
    # args.model_path, args.worker_host, args.worker_port = item.split("@")
    print("*" * 80)
    worker_str_args = f" --model {model} {worker_str_args}"
    print(worker_str_args)
    # "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"
    worker_sh = base_launch_sh.format(
        "llm_model/hp_worker_launcher.py", worker_str_args, LOGDIR, f"worker_{log_name}"
    )
    worker_sh = get_gpus(model_name) + worker_sh
    worker_check_sh = base_check_model_sh.format(int(wait_times / 2), LOGDIR,
                                                 f"worker_{log_name}",
                                                 "model_worker",
                                                 model_name)
    print(f"executing worker_sh: {worker_sh}")
    subprocess.run(worker_sh, shell=True, check=True)
    try:
        subprocess.run(worker_check_sh, shell=True, check=True)
        time.sleep(1)
        return True, None
    except subprocess.CalledProcessError as e:
        print(e)
        return False, f"start {model} worker fail, see the log for details"


def launch_api_server(server_str_args: str = ""):
    server_sh = base_launch_sh.format(
        "llm_chat/hp_api_server_launcher.py", server_str_args, LOGDIR, "api_server"
    )
    server_sh = "cd ../jian && " + server_sh
    server_check_sh = base_check_sh.format(10, LOGDIR, "api_server", "api_server")
    subprocess.run(server_sh, shell=True, check=True)
    subprocess.run(server_check_sh, shell=True, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers",
        dest="all",
    )
    parser.add_argument(
        "-c",
        "--controller",
        action="store_true",
        help="run fastchat's controller servers",
        dest="controller",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="run fastchat's openai_api servers",
        dest="openai_api",
    )
    parser.add_argument(
        "-w",
        "--model-worker",
        action="store_true",
        help="run fastchat's model_worker server with specified model name. "
             "specify -m if not using default LLM_MODELS",
        dest="model_worker",
    )

    parser.add_argument(
        "-cp",
        "--controller-config-path",
        help="custom controller config path",
        dest="cp",
        default=None,
    )
    controller_args = ["cp", "verbose"]

    openaiapi_server_args = ["verbose"]

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
        "-turbo",
        "--infer-turbo",
        help="custom model worker infer turbo",
        dest="infer_turbo",
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
    model_worker_args = ["wp", "infer-turbo", "verbose"]

    # ---------------------------------------------MAIN---------------------------------------------------
    args = parser.parse_args()


    def launch_all():
        if args.all:
            args.controller = True
            args.openai_api = True
            args.model_worker = True

        if args.controller:
            controller_str_args = string_args(args, controller_args)
            # "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"
            controller_sh = base_launch_sh.format(
                "llm_model/hp_controller_launcher.py", controller_str_args, LOGDIR, "controller"
            )
            controller_check_sh = base_check_sh.format(10, LOGDIR, "controller", "controller")
            print(f"executing controller_sh: {controller_sh}")
            # print(f"watch controller log: {controller_check_sh}")
            subprocess.run(controller_sh, shell=True, check=True)
            subprocess.run(controller_check_sh, shell=True, check=True)

        if args.model_worker and args.model:
            worker_str_args = string_args(args, model_worker_args)
            if isinstance(args.model, str):
                launch_worker(args.model, worker_str_args)
            else:
                for idx, item in enumerate(args.model):
                    print(f"loading {idx}th model:{item}")
                    launch_worker(item, worker_str_args)

        if args.openai_api:
            server_str_args = string_args(args, openaiapi_server_args)
            launch_api_server(server_str_args)


    launch_all()
