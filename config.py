from typing import Dict
from enum import Enum

class ConfigKey(Enum):
    FASTEMBED_CACHE = "./fastembed"
    DB_HOST = "http://localhost:6333"
    LLM_HOST = "http://127.0.0.1:8080"
    BENCHMARK = False
    # add more as needed...

_config: Dict[str, any] = {}

def get(k: ConfigKey) -> any:
    if _config.get(k.name) is None:
        set(k)
    
    return _config[k.name]

def set(k: ConfigKey, v: any=None):
    if v is not None:
        _config[k.name] = v
    else:
        _config[k.name] = k.value
