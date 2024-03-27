from typing import List, Literal, Optional, Dict
from fuxi.utils.torch_helper import detect_device

from dynaconf import Dynaconf

cfg: Dynaconf = None

OPENAI_EMBEDDINGS_CHUNK_SIZE = 500

# def embedding_device(device: str = None) -> str:
#     device: str = device or EMBEDDING_DEVICE
#     if device not in ["cuda", "mps", "cpu"]:
#         device = detect_device()
#     return device


def get_default_embed_model():
    return cfg.get("embed.default_run", ["bge-large-zh-v1.5"])


def get_default_embed_device():
    return cfg.get("embed.device")


# 从model_config中获取模型信息

def get_config_embed_models() -> Dict:
    return cfg.get("embed.model_cfg", {})


def get_embed_model_path(model_name: str) -> str:
    return get_config_embed_models()[model_name]

# online_embed_models = {}
#
#
# def get_online_embed_models() -> Dict:
#     return online_embed_models
