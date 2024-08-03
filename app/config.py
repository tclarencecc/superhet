import uuid
import sys
from argparse import ArgumentParser
import yaml

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
            "QDRANT__STORAGE__STORAGE_PATH": "" # from config
        }
    QDRANT = _qdrant

    class _llama:
        class _completion:
            MODEL = "" # from config
            TEMPERATURE = 0.1
            FLASH_ATTENTION = False # from config

            # add more as needed...
        COMPLETION = _completion

        class _embedding:
            # all from config
            MODEL = ""
            CONTEXT = 0
            SIZE = 0
        EMBEDDING = _embedding
    LLAMA = _llama

    class _chunk:
        SIZE = _min_max(20, None) # no MAX; computed from EMBEDDING.CONTEXT
        OVERLAP = _min_max(0.1, 0.25)
    CHUNK = _chunk

    BENCHMARK = not in_prod()

    PROCESS_STDOUT = False

    # qdrant recommends multitenancy as opposed to multicollection
    COLLECTION = "my collection"

    CLI_CMD_PREFIX = "!"

    # add more as needed...


if in_prod():
    config_path = "./config.yaml" # default same dir as executable

    parser = ArgumentParser()
    parser.add_argument("exe") # ./main itself, just ignore
    parser.add_argument("-cfg", "--config", type=str, required=False)
    arg = parser.parse_args(sys.argv)

    if arg.config is not None:
        config_path = arg.config
else:
    config_path = "../config.yaml" # outside project folder

try:
    with open(config_path) as f:
        # any of these can raise error
        obj = yaml.safe_load(f)

        db_path = obj["db"]["path"]

        llm_c_model = obj["llm"]["completion"]["model"]
        llm_c_fa = obj["llm"]["completion"]["flash_attention"]

        llm_e_model = obj["llm"]["embedding"]["model"]
        llm_e_context = obj["llm"]["embedding"]["context"]
        llm_e_size = obj["llm"]["embedding"]["size"]

        # reaching here means yaml object is valid, set values into Config
        Config.QDRANT.ENV["QDRANT__STORAGE__STORAGE_PATH"] = db_path

        Config.LLAMA.COMPLETION.MODEL = llm_c_model
        Config.LLAMA.COMPLETION.FLASH_ATTENTION = llm_c_fa

        Config.LLAMA.EMBEDDING.MODEL = llm_e_model
        Config.LLAMA.EMBEDDING.CONTEXT = llm_e_context
        Config.LLAMA.EMBEDDING.SIZE = llm_e_size

except IOError:
    print("Config file not found")
    sys.exit()
except (yaml.YAMLError, KeyError):
    print("Config file invalid")
    sys.exit()

# argv overrides config
# if in_prod():
#     Config.XYZ = arg.xyz
