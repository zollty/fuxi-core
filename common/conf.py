
import tomli
from common.utils import detect_device
from typing import (Any)

class Cfg():
    toml_dict: dict[str, Any]
    llm_device: str
    embedding_device: str

    def __init__(self, conf_path: str):
        with open(conf_path, "rb") as f:
            try:
                self.toml_dict = tomli.load(f)
                print(self.toml_dict)
            except tomli.TOMLDecodeError:
                print("Yep, definitely not valid.")
       
        self.embedding_device = self.get("embed.device")
        self.llm_device = self.get("llm.device")
        if self.llm_device == "auto" or self.embedding_device == "auto":
            _auto = detect_device()
            if self.llm_device == "auto":
                self.llm_device = _auto
            if  self.embedding_device == "auto":
                self.embedding_device = _auto


    def get(self, key: str, default_val = None):
        sv = key.split(".")
        le = len(sv)
        print(f"get key: {sv}")
        ret = None
        try:
            if le == 1:
                ret = self.toml_dict[sv[0]]
            elif le == 2:
                ret = self.toml_dict[sv[0]][sv[1]]
            elif le == 3:
                ret = self.toml_dict[sv[0]][sv[1]][sv[2]]
            elif le == 4:
                ret = self.toml_dict[sv[0]][sv[1]][sv[2]][sv[3]]
            elif le == 5:
                ret = self.toml_dict[sv[0]][sv[1]][sv[2]][sv[3]][sv[4]]
            elif le == 6:
                ret = self.toml_dict[sv[0]][sv[1]][sv[2]][sv[3]][sv[4]][sv[5]]
            elif le == 7:
                ret = self.toml_dict[sv[0]][sv[1]][sv[2]][sv[3]][sv[4]][sv[5]][sv[6]]
            elif le == 8:
                ret = self.toml_dict[sv[0]][sv[1]][sv[2]][sv[3]][sv[4]][sv[5]][sv[6]][sv[7]]
            elif le == 9:
                ret = self.toml_dict[sv[0]][sv[1]][sv[2]][sv[3]][sv[4]][sv[5]][sv[6]][sv[7]][sv[8]]
        except:
            print("this key not exist!")
        if default_val == None:
            return ret
        elif ret:
            return ret
        else:
            return default_val
