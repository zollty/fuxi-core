from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Callable,
    Generator,
    Dict,
    Any,
    Awaitable,
    Union,
    Tuple
)

import os
# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
get_runtime_root_dir() = os.path.dirname(os.path.dirname(__current_script_path))

# 日志存储路径
DEFAULT_LOG_PATH = os.path.join(get_runtime_root_dir(), "logs")
if not os.path.exists(DEFAULT_LOG_PATH):
    os.mkdir(DEFAULT_LOG_PATH)

VERSION = "1.0.0"
# API 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = True
# SSL_KEYFILE = os.environ["SSL_KEYFILE"]
# SSL_CERTFILE = os.environ["SSL_CERTFILE"]

import logging

# 日志格式
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
# 是否显示详细日志
LOG_VERBOSE = False
# 通常情况下不需要更改以下内容

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

# NLTK模型分词模型 例如：NLTKTextSplitter，SpacyTextSplitter，配置nltk 模型存储路径
# import nltk
# NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
# nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"

def decide_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    if not device or device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device