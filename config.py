from typing import Dict
from enum import Enum

class ConfigKey(Enum):
    FASTEMBED_CACHE = "./fastembed"
    DB_HOST = "http://localhost:6333"
    LLM_HOST = "http://127.0.0.1:8080"

_config: Dict[str, str] = {}

def get(k: ConfigKey) -> str:
    if _config.get(k.name) is None:
        set(k)
    
    return _config[k.name]

def set(k: ConfigKey, v: str=None) -> None:
    if v is not None and v != "":
        _config[k.name] = v
    else:
        _config[k.name] = k.value
