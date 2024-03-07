"""
Usage：
python shutdown_serve.py --down all
options: "all","controller","model_worker","openai_api_server"， `all` means to stop all related servers 
"""
import subprocess

def shutdown_serve(key):
    base_shell = "ps -eo user,pid,cmd|grep fschat_{}|grep -v grep|awk '{{print $2}}'|xargs kill -9"
    if key == "all":
        shell_script = base_shell.format("")
    else:
        shell_script = base_shell.format(key)
    print(f"execute shell cmd: {shell_script}")
    subprocess.run(shell_script, shell=True, check=True)
    print(f"{key} has been shutdown!")

def shutdown_worker_serve(model):
    base_shell = "ps -eo user,pid,cmd|grep fschat_worker_launcher.py|grep {}|grep -v grep|awk '{{print $2}}'|xargs kill -9"
    shell_script = base_shell.format(model)
    print(f"execute shell cmd: {shell_script}")
    subprocess.run(shell_script, shell=True, check=True)
    print(f"{model} has been shutdown!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--down", choices=["all", "controller", "worker", "openai", "c", "w", "o"]
    )
    parser.add_argument(
        "--model", type=str, default=None
    )
    args = parser.parse_args()

    if args.model:
        shutdown_worker_serve(args.model)
    else:
        shutdown_serve(args.down)
