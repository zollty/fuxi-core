import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
get_runtime_root_dir() = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(get_runtime_root_dir())
print(get_runtime_root_dir())

import numpy as np

x = np.random.rand(10)
print("hello")
print(x)

import tomli

toml_str = """
[[players]]
name = "Lehtinen"
number = 26

[[players]]
name = "Numminen"
number = 27
"""




with open("./config.toml", "rb") as f:
    try:
        toml_dict = tomli.load(f)
    except tomli.TOMLDecodeError:
        print("Yep, definitely not valid.")
    print(toml_dict)
    # assert toml_dict == {
    #     "players": [{"name": "Lehtinen", "number": 26}, {"name": "Numminen", "number": 27}]
    # }
    toml_dict["servers"]["alpha"]["ip"] = "sssssssssss"
    print(toml_dict["servers"]["alpha"]["ip"])

    toml_dict["database"]["data"][0].append("ssssslll")
    print(toml_dict["database"]["data"][0])

    print(toml_dict["demo"]["re"])
    print(toml_dict["demo"]["lines"])
    print(toml_dict["demo"]["regex"])
    print(toml_dict["demo"].get("regex9", "==========================="))


toml_dict = tomli.loads(toml_str)
assert toml_dict == {
    "players": [{"name": "Lehtinen", "number": 26}, {"name": "Numminen", "number": 27}]
}

from decimal import Decimal
toml_dict = tomli.loads("precision-matters = 0.982492", parse_float=Decimal)
assert toml_dict["precision-matters"] == Decimal("0.982492")


FSCHAT_MODEL_WORKERS = {
    # 所有模型共用的默认配置，可在模型专项配置中进行覆盖。
    "aaa": {
        "host": "DEFAULT_BIND_HOST",
        "port": 20002,
        "device": "LLM_DEVICE",
    }
}

print(FSCHAT_MODEL_WORKERS["aaa"])