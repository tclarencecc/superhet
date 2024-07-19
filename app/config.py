import uuid

_llama_key = uuid.uuid4().hex
_qdrant_key = uuid.uuid4().hex

class _min_max:
    def __init__(self, min, max): # no type; can be int or float
        self.MIN = min
        self.MAX = max

class Config:
    class _fastembed:
        PATH = "./bin"
        TOKEN = 512
    FASTEMBED = _fastembed

    class _qdrant:
        HOST = "http://localhost:6333"
        PATH = "./bin"
        # config.yaml is only for setting grpc_port=null as its impossible to pass null/None in env dict
        SHELL = "./qdrant --config-path ./config.yaml"
        KEY = _qdrant_key
        # https://qdrant.tech/documentation/guides/configuration/#environment-variables
        ENV = {
            "QDRANT__SERVICE__API_KEY": _qdrant_key,
            "QDRANT__TELEMETRY_DISABLED": "true"
        }
    QDRANT = _qdrant

    class _llama:
        HOST = "http://127.0.0.1:8080"
        PATH = "./bin"
        SHELL = "./llama-server -m ./qwen2-1_5b-instruct-q8_0.gguf -fa -n 256 --log-disable --api-key " + _llama_key
        KEY = _llama_key
    LLAMA = _llama

    class _chunk:
        SIZE = _min_max(20, None) # no MAX; computed from FASTEMBED.TOKEN
        OVERLAP = _min_max(0.1, 0.25)
    CHUNK = _chunk

    BENCHMARK = True

    PROCESS_STDOUT = False

    # qdrant recommends multitenancy as opposed to multicollection
    COLLECTION = "my collection"

    # add more as needed...
