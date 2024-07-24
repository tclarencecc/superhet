import uuid
import sys
import yaml

_llama_key = uuid.uuid4().hex
_qdrant_key = uuid.uuid4().hex

def in_prod() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

def _select(dev: str, prod: str) -> str:
    if in_prod():
        return prod
    else:
        return dev
    
def _binary_path() -> str:
    return _select("./bin", "./_internal")

class _min_max:
    def __init__(self, min, max): # no type; can be int or float
        self.MIN = min
        self.MAX = max

class Config:
    class _fastembed:
        PATH = _binary_path()
        TOKEN = 512
    FASTEMBED = _fastembed

    class _qdrant:
        HOST = "http://localhost:6333"
        GRPC = 6334
        PATH = _binary_path()
        SHELL = "./qdrant"
        KEY = _qdrant_key
        READ_LIMIT = 1
        # https://qdrant.tech/documentation/guides/configuration/#environment-variables
        ENV = {
            "QDRANT__SERVICE__API_KEY": _qdrant_key,
            "QDRANT__TELEMETRY_DISABLED": "true",
            "QDRANT__STORAGE__STORAGE_PATH": "../../db" # dev mode default outside project folder
        }
    QDRANT = _qdrant

    class _llama:
        HOST = "http://127.0.0.1:8080"
        PATH = _binary_path()
        MODEL = "../../qwen2-1_5b-instruct-q4_k_m.gguf" # dev mode default outside project folder
        KEY = _llama_key

        class _option:
            TEMPERATURE = 0.1
            # add more completion option as needed...
        OPTION = _option

        @staticmethod
        def get_shell() -> str:
            return f"./llama-server -m {Config.LLAMA.MODEL} -fa --log-disable --api-key {_llama_key}"
        SHELL = "" # generate on runtime using get_shell!
    LLAMA = _llama

    class _chunk:
        SIZE = _min_max(20, None) # no MAX; computed from FASTEMBED.TOKEN
        OVERLAP = _min_max(0.1, 0.25)
    CHUNK = _chunk

    BENCHMARK = not in_prod()

    PROCESS_STDOUT = False

    # qdrant recommends multitenancy as opposed to multicollection
    COLLECTION = "my collection"

    CLI_CMD_PREFIX = "!"

    # add more as needed...


if in_prod():
    try:
        with open("./config.yaml") as f:
            # any of these can raise error
            obj = yaml.safe_load(f)
            db_path = obj["db"]["path"]
            llm_model = obj["llm"]["model"]

            # reaching here means yaml object is valid, set values into Config
            Config.QDRANT.ENV["QDRANT__STORAGE__STORAGE_PATH"] = db_path
            Config.LLAMA.MODEL = llm_model
    except IOError:
        print("Config file not found")
    except (yaml.YAMLError, KeyError):
        print("Config file invalid")

Config.LLAMA.SHELL = Config._llama.get_shell()
