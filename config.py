from typing import Dict
from enum import Enum

class ConfigKey(Enum):
    FASTEMBED = {
        "name": "bge-small-en-v1.5",
        "path": "./fastembed",
        "token": 512
    }
    DB_HOST = "http://localhost:6333"
    LLM_HOST = "http://127.0.0.1:8080"
    BENCHMARK = False
    CHUNK = {
        "minimum": 20, # a sentence basically. max is computed from FASTEMBED.token
        "overlap": [0.1, 0.25]
    }
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
