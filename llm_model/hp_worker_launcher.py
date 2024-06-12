import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
runtime_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__current_script_path)))
sys.path.append(runtime_root_dir)

from dynaconf import Dynaconf
from fastapi import FastAPI
from fuxi.utils.torch_helper import detect_device

from fuxi.utils.runtime_conf import logger


def set_common_args(args):
    if not args.get("controller_address"):
        args["controller_address"] = args["controller_addr"]
    if not args.get("worker_address"):
        args["worker_address"] = args["worker_addr"]
    if args["device"] == "auto":
        args["device"] = detect_device()
    if args.get("gpus"):
        if args.get("num_gpus") is None:
            args["num_gpus"] = len(args.gpus.split(','))
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def create_vllm_worker(cfg: Dynaconf, model_worker_config, log_level):
    from fastchat.serve.vllm_worker import VLLMWorker, app, worker_id
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    import argparse

    vllm_args = cfg.get("llm.worker.base") + cfg.get("llm.worker.vllm") + model_worker_config.get("base")
    if model_worker_config.get("vllm"):
        vllm_args = vllm_args + model_worker_config.get("vllm")

    set_common_args(vllm_args)

    vllm_args["tokenizer"] = vllm_args["model_path"]

    if vllm_args.model_path:
        vllm_args.model = vllm_args.model_path
    # if vllm_args.num_gpus > 1:
    #     vllm_args.tensor_parallel_size = vllm_args.num_gpus

    parser = argparse.ArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    # args = parser.parse_args([])
    args = parser.parse_args()
    for k, v in vllm_args.items():
        setattr(args, k, v)
    logger.info("---------------------vllm_args------------------------")
    logger.info(vllm_args)
    logger.info("---------------------vllm_args------------------------")

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    worker = VLLMWorker(
        controller_addr=args.controller_addr,
        worker_addr=args.worker_addr,
        worker_id=worker_id,
        model_path=args.model_path,
        model_names=args.model_names,
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
        llm_engine=engine,
        conv_template=args.conv_template,
    )
    sys.modules["fastchat.serve.vllm_worker"].engine = engine
    sys.modules["fastchat.serve.vllm_worker"].worker = worker
    sys.modules["fastchat.serve.vllm_worker"].logger.setLevel(log_level)

    return app, worker


def create_sglang_worker(cfg: Dynaconf, model_worker_config, log_level: str):
    from fastchat.serve.sglang_worker import SGLWorker, app, worker_id
    import multiprocessing
    import sglang as sgl
    from sglang.srt.hf_transformers_utils import get_tokenizer, get_config
    from sglang.srt.utils import load_image, is_multimodal_model

    import argparse

    sglang_args = cfg.get("llm.worker.base") + cfg.get("llm.worker.sglang") + model_worker_config.get("base")
    if model_worker_config.get("sglang"):
        sglang_args = sglang_args + model_worker_config.get("sglang")

    set_common_args(sglang_args)

    if not sglang_args.get("tokenizer_path"):
        sglang_args["tokenizer_path"] = sglang_args["model_path"]

    if sglang_args.num_gpus > 1:
        sglang_args["tp_size"] = sglang_args.num_gpus

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    for k, v in sglang_args.items():
        setattr(args, k, v)
    logger.info("---------------------vllm_args------------------------")
    logger.info(sglang_args)
    logger.info("---------------------vllm_args------------------------")

    multiprocessing.set_start_method("spawn", force=True)
    runtime = sgl.Runtime(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        mem_fraction_static=args.mem_fraction_static,
        tp_size=args.tp_size,
        log_level=log_level.lower(),
    )
    sgl.set_default_backend(runtime)

    worker = SGLWorker(
        args.controller_addr,
        args.worker_addr,
        worker_id,
        args.model_path,
        args.tokenizer_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        runtime,
        args.trust_remote_code,
    )

    sys.modules["fastchat.serve.sglang_worker"].worker = worker
    sys.modules["fastchat.serve.sglang_worker"].logger.setLevel(log_level)

    return app, worker


def create_plain_worker(cfg: Dynaconf, model_worker_config, log_level):
    from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker, worker_id
    gptq_args = None
    awq_args = None
    args = cfg.get("llm.worker.base") + cfg.get("llm.worker.plain") + model_worker_config.get("base")
    if model_worker_config.get("plain"):
        args = args + model_worker_config.get("plain")
        gptq_args = model_worker_config["plain"].get("gptq")
        awq_args = model_worker_config["plain"].get("awq")
    set_common_args(args)
    logger.info("---------------------worker_args------------------------")
    logger.info(args)

    if not gptq_args:
        gptq_args = cfg.get("llm.worker.plain.gptq")
    elif cfg.get("llm.worker.plain.gptq"):
        gptq_args = cfg.get("llm.worker.plain.gptq") + gptq_args
    gptq_config = None
    if gptq_args:
        logger.info("---------------------gptq_args------------------------")
        logger.info(gptq_args)
        gptq_config = GptqConfig(
            ckpt=gptq_args.gptq_ckpt or args.model_path,
            wbits=gptq_args.gptq_wbits,
            groupsize=gptq_args.gptq_groupsize,
            act_order=gptq_args.gptq_act_order,
        )

    if not awq_args:
        awq_args = cfg.get("llm.worker.plain.awq")
    elif cfg.get("llm.worker.plain.awq"):
        awq_args = cfg.get("llm.worker.plain.awq") + awq_args
    awq_config = None
    if awq_args:
        logger.info("---------------------awq_args------------------------")
        logger.info(awq_args)
        awq_config = AWQConfig(
            ckpt=awq_args.awq_ckpt or args.model_path,
            wbits=awq_args.awq_wbits,
            groupsize=awq_args.awq_groupsize,
        )

    worker = ModelWorker(
        controller_addr=args.controller_addr,
        worker_addr=args.worker_addr,
        worker_id=worker_id,
        model_path=args.model_path,
        model_names=args.model_names,
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
    )
    sys.modules["fastchat.serve.model_worker"].args = args
    sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
    # sys.modules["fastchat.serve.model_worker"].worker = worker
    sys.modules["fastchat.serve.model_worker"].logger.setLevel(log_level)

    return app, worker


def create_worker_app(cfg: Dynaconf, model_worker_config, log_level) -> FastAPI:
    """
    kwargs包含的字段如下：
    host:
    port:
    model_names:[`model_name`]
    controller_addr:
    worker_addr:

    对于Langchain支持的模型：
        langchain_model:True
        不会使用fschat
    对于online_api:
        online_api:True
        worker_class: `provider`
    对于离线模型：
        model_path: `model_name_or_path`,huggingface的repo-id或本地路径
        device:`LLM_DEVICE`
    """
    from fuxi.utils.runtime_conf import get_default_log_path
    from fuxi.utils.fastapi_tool import MakeFastAPIOffline
    import sys
    import fastchat.constants
    from fastchat.serve.model_worker import logger
    from fastapi.middleware.cors import CORSMiddleware

    fastchat.constants.LOGDIR = get_default_log_path()
    log_level = log_level.upper()
    logger.setLevel(log_level)

    model_name = model_worker_config.get("model_name")
    worker_port = model_worker_config.get("port")
    if not worker_port:
        start_port = cfg.get("llm.worker.start_port")
        worker_port = start_port
        model_worker_config["port"] = worker_port

    host = cfg.get("llm.worker.host")
    worker_addr = f"http://{host}:{worker_port}"
    model_worker_config["base"]["worker_addr"] = worker_addr
    model_worker_config["base"]["model_path"] = model_worker_config.get("path")
    model_worker_config["base"]["model_names"] = [model_worker_config.get("model_name")]

    if model_worker_config.get("langchain_model"):  # Langchian支持的模型不用做操作
        from fastchat.serve.base_model_worker import app
        worker = None
    # 在线模型API
    elif worker_class := model_worker_config.get("worker_class"):
        from fastchat.serve.base_model_worker import app

        worker = worker_class(model_names=[model_name],
                              controller_addr=cfg.llm.worker.base.controller_addr,
                              worker_addr=worker_addr)
        # sys.modules["fastchat.serve.base_model_worker"].worker = worker
        sys.modules["fastchat.serve.base_model_worker"].logger.setLevel(log_level)
    # 本地模型
    else:
        if model_worker_config.get("infer_turbo") == "vllm":
            app, worker = create_vllm_worker(cfg, model_worker_config, log_level)

        elif model_worker_config.get("infer_turbo") == "sglang":
            app, worker = create_sglang_worker(cfg, model_worker_config, log_level)

        else:
            app, worker = create_plain_worker(cfg, model_worker_config, log_level)

    cross_domain = cfg.get("llm.worker.cross_domain", cfg.get("root.cross_domain", True))

    if cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.title = f"伏羲AI LLM Worker Server ({model_name})"
    app.version = fastchat.__version__
    app._worker = worker
    MakeFastAPIOffline(app)
    return app


def run_model_worker(model_name, port: str = None):
    from fuxi.utils.runtime_conf import get_runtime_root_dir
    from fuxi.utils.fastapi_tool import run_api

    print(get_runtime_root_dir())

    from dynaconf import Dynaconf

    cfg = Dynaconf(
        envvar_prefix="HP",
        root_path=get_runtime_root_dir(),
        settings_files=['conf/llm_model.yml', 'conf/settings.yaml'],  # 后者优先级高，以一级key覆盖前者（一级key相同的，前者不生效）
    )

    log_level = cfg.get("llm.worker.log_level", cfg.get("root.log_level", "INFO"))

    model_worker_config = {"model_name": model_name}
    if model_name == "langchain_model":
        model_worker_config["langchain_model"] = True
    else:
        model_worker_config = cfg.get("llm.model_cfg")[model_name] + model_worker_config

    if port and int(port) > 1000:
        model_worker_config["port"] = int(port)

    from fuxi.utils.fastapi_tool import set_httpx_config
    set_httpx_config()

    app = create_worker_app(cfg, model_worker_config, log_level)

    host = cfg.get("llm.worker.host")
    use_port = model_worker_config.get("port")

    # server info
    with open(get_runtime_root_dir() + '/logs/start_info.txt', 'a') as f:
        f.write(f"    FenghouAI Model Worker Server ({model_name}): http://{host}:{use_port}")

    if host == "localhost" or host == "127.0.0.1":
        host = "0.0.0.0"
    run_api(
        app,
        host=host,
        port=use_port,
        log_level=log_level,
        ssl_keyfile=cfg.get("llm.worker.ssl_keyfile", cfg.get("root.ssl_keyfile")),
        ssl_certfile=cfg.get("llm.worker.ssl_certfile", cfg.get("root.ssl_certfile")),
    )


# CUDA_VISIBLE_DEVICES=$gpuid1 nohup python3 -m fastchat.serve.model_worker --host 0.0.0.0 --port 7801 \
#     --controller-address http://0.0.0.0:7800 --worker-address http://0.0.0.0:7801 \
#     --model-path /model_repos/QAEnsemble/$LLM_API_SERVE_MODEL --load-8bit \
#     --dtype bfloat16 --conv-template $LLM_API_SERVE_CONV_TEMPLATE > fschat_model_worker_7801.log 2>&1 &
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    # parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--custom-config", type=str, default=None)
    parser.add_argument(
        "-v",
        "--verbose",
        help="增加log信息",
        dest="verbose",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    ret = args.model.split("@")
    model_name = args.model
    port = None
    if len(ret) == 2:
        model_name, port = ret

    # run_worker("langchain_model")
    # run_worker("chatglm3-6b-32k")
    # run_model_worker("Qwen1.5-7B-Chat")
    run_model_worker(model_name, port)
