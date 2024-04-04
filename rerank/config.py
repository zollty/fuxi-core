from typing import List, Literal, Optional, Dict
from fuxi.utils.torch_helper import detect_device

from dynaconf import Dynaconf

cfg: Dynaconf = None


# def embedding_device(device: str = None) -> str:
#     device: str = device or EMBEDDING_DEVICE
#     if device not in ["cuda", "mps", "cpu"]:
#         device = detect_device()
#     return device


def get_default_rerank_model() -> list[str]:
    return cfg.get("rerank.default_run", ["bge-large-zh-v1.5"])


def get_default_rerank_device() -> str:
    return cfg.get("rerank.device")


def get_default_rerank_num_workers() -> int:
    return cfg.get("rerank.num_workers")


def get_default_rerank_batch_size() -> int:
    return cfg.get("rerank.batch_size")


# 从model_config中获取模型信息

def get_config_rerank_models() -> Dict:
    return cfg.get("rerank.model_cfg", {})


def get_rerank_model_path(model_name: str) -> str:
    return get_config_rerank_models()[model_name]

# online_embed_models = {}
#
#
# def get_online_embed_models() -> Dict:
#     return online_embed_models
