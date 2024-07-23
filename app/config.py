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
        GRPC = 6334
        PATH = "./bin"
        SHELL = "./qdrant"
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
        SHELL = f"./llama-server -m ./qwen2-1_5b-instruct-q4_k_m.gguf -fa --log-disable --api-key {_llama_key}"
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

    CLI_CMD_PREFIX = "!"

    # add more as needed...
