import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
RUNTIME_ROOT_DIR = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(RUNTIME_ROOT_DIR)

toml_str = """
[embed]
device = "cuda"

[[players]]
name = "Lehtinen"
number = 26

[[players]]
name = "Numminen"
number = 27
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class KbConfig:
    # 默认使用的知识库
    DEFAULT_KNOWLEDGE_BASE = "samples"
    # kbs_config: Dict[str, Dict] = field(default_factory=lambda:
    kbs_config = {
        "faiss": {
        },
        "milvus": {
            "host": "127.0.0.1",
            "port": "19530",
            "user": "",
            "password": "",
            "secure": False,
        },
        "zilliz": {
            "host": "in01-a7ce524e41e3935.ali-cn-hangzhou.vectordb.zilliz.com.cn",
            "port": "19530",
            "user": "",
            "password": "",
            "secure": True,
        },
        "pg": {
            "connection_uri": "postgresql://postgres:postgres@127.0.0.1:5432/fenghou-ai",
        },

        "es": {
            "host": "127.0.0.1",
            "port": "9200",
            "index_name": "test_index",
            "user": "",
            "password": ""
        },
        "milvus_kwargs": {
            "search_params": {"metric_type": "L2"},  # 在此处增加search_params
            "index_params": {"metric_type": "L2", "index_type": "HNSW"}  # 在此处增加index_params
        },
        "chromadb": {}
    }


if __name__ == "__main__":
    from common.conf import Cfg
    from common.utils import RUNTIME_ROOT_DIR

    print(RUNTIME_ROOT_DIR)
    cfg = Cfg(RUNTIME_ROOT_DIR + "/conf_rerank_test.toml")
    print(cfg.get("reranker.model.bge-reranker-large"))
    print(cfg.get("embed.device"))

    print("---------222------------------")
    cfg = Cfg(toml_str, False, None)
    print(cfg.get("players[1].name"))


    # print(cfg.get("servers.alpha.ip"))

    def props(obj):
        pr = {}
        for name in dir(obj):
            value = getattr(obj, name)
            if not name.startswith('__') and not callable(value):
                pr[name] = value
        return pr


    # from embed.server_config import ServerConfig
    # conf = props(ServerConfig)
    # print(conf)
    from omegaconf import OmegaConf

    # conf = OmegaConf.structured(KbConfig)
    # print(OmegaConf.to_yaml(conf))
    conf = props(KbConfig)
    print(conf)
    conf = OmegaConf.create(conf)
    print(conf.kbs_config)

    conf = OmegaConf.load(RUNTIME_ROOT_DIR + '/llm_model/conf_llm_model.yml')
    print(conf.llm.model_cfg)
    for mc in conf["llm"]["model_cfg"].items():
        print(mc)

    from dynaconf import Dynaconf

    cfg = Dynaconf(
        envvar_prefix="FUXI",
        root_path=RUNTIME_ROOT_DIR,
        settings_files=['llm_model/conf_llm_model.yml', 'settings.yaml'],
    )

    print("===================================")
    print(cfg["test-aa.key-bb"])
    print(cfg.get("test-aa.key-bb"))
    print(cfg.get("test-aa.key-bb-cc", cfg.get("llm.worker.base.controller_addr")))
    for k, v in cfg.items():
        print(k, v)

    tgf = cfg.get("llm.worker.vllm")
    tgf["cc"] = "dsjhdhjsjhds"
    print(tgf)
    print(cfg.get("llm.worker.vllm"))

    tfg1 = cfg.get("llm.worker.base") + {}
    print("-------------------------")
    tfg1["cc"] = "xxxxxxxxx"
    print(getattr(tfg1, "conv_template"))

    print(cfg.get("llm.worker.base"))
    print(cfg.get("llm.worker.vllm"))

    cc = {"sss": 123}
    dd = None
    if cc:
        print(cc + dd)
