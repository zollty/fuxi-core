
import tomli
from common.utils import detect_device
from typing import (Any)

class Cfg():
    '''
    llm.device
    embed.device
    '''
    toml_dict: dict[str, Any]

    def __init__(self, conf_path: str):
        with open(conf_path, "rb") as f:
            try:
                self.toml_dict = tomli.load(f)
                print(self.toml_dict)
            except tomli.TOMLDecodeError:
                print("Yep, definitely not valid.")
       
        if self.get("llm.device") == "auto":
            _auto = detect_device()
            self.set("llm.device", _auto)

        if self.get("embed.device") == "auto":
            if _auto is None:
                _auto = detect_device()
            self.set("embed.device", _auto)

    def embedding_device(self, default_val = None):
        if self.embedding_device is None:
            return default_val
        return self.embedding_device
    
    def set(self, key: str, val = None):
        self.toml_dict[key] = val

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
