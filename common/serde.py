import json
import inspect
from enum import Enum

class SerdeParseError(Exception):
    ...

class Serde:
    # only call super.__init__() after initializing child class attributes!
    def __init__(self, type: Enum, json_str: str):
        self.type = type

        if json_str != "":
            try:
                obj: dict = json.loads(json_str)
            except:
                raise SerdeParseError(f"Parsing error: {json_str}")
            
            keys = set()
            for k in obj.keys():
                keys.add(k)

            if not (keys == self._attributes()): # set == compares similarity of ALL elements
                raise SerdeParseError(f"Json schema mismatch: {json_str}")

            for k in keys:
                v = obj[k]
                if k == "type":
                    v = type.__class__[v]
                setattr(self, k, v)

    def _attributes(self) -> set[str]:
        ret = set()
        for mbr in inspect.getmembers(self):
            if mbr[0].startswith(("__", "_")):
                continue
            if str(type(mbr[1])) == "<class 'method'>":
                continue

            ret.add(mbr[0])
        return ret
    
    def json(self) -> dict:
        ret = {}
        for k in self._attributes():
            v = getattr(self, k)
            if k == "type":
                v = v.name
            ret[k] = v
        return ret
    
    def json_string(self) -> str:
        return json.dumps(self.json())

def parse_type(json_str: str, enum_t: Enum) -> Enum:
    try:
        obj: dict = json.loads(json_str)
        return enum_t[obj["type"]]
    except:
        raise SerdeParseError(f"Parsing error: {json_str}")
