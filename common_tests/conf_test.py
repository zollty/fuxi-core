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
