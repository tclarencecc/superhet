import uuid
import sys
from argparse import ArgumentParser
from enum import Enum

from app.util import Toml

_qdrant_key = uuid.uuid4().hex

def in_prod() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

def _select(dev: str, prod: str) -> str:
    if in_prod():
        return prod
    else:
        return dev
    
class _min_max:
    def __init__(self, min, max): # no type; can be int or float
        self.MIN = min
        self.MAX = max

class PromptFormat(Enum):
    CHATML = 1,
    GEMMA = 2,
    LLAMA = 3

class Config:
    class _qdrant:
        HOST = "http://localhost:6333"
        PATH = _select("./bin", "./_internal")
        SHELL = "./qdrant"
        KEY = _qdrant_key
        READ_LIMIT = 1
        # https://qdrant.tech/documentation/guides/configuration/#environment-variables
        ENV = {
            "QDRANT__SERVICE__API_KEY": _qdrant_key,
            "QDRANT__TELEMETRY_DISABLED": "true",
            "QDRANT__STORAGE__STORAGE_PATH": None, # manually parsed from toml
            "QDRANT__STORAGE__SNAPSHOTS_PATH": None # derived from STORAGE_PATH
        }
    QDRANT = _qdrant

    class _llama:
        class _completion:
            MODEL = Toml.Spec("llm.completion.model")
            PROMPT_FORMAT = Toml.Spec("llm.completion.prompt_format", None, lambda x: PromptFormat[x])
            FLASH_ATTENTION = Toml.Spec("llm.completion.flash_attention", False)
            CONTEXT_SIZE = Toml.Spec("llm.completion.context_size", 0)

            # hardcoded
            TEMPERATURE = 0
        COMPLETION = _completion

        class _embedding:
            MODEL = Toml.Spec("llm.embedding.model")

            # derived from gguf metadata
            CONTEXT = None
            SIZE = None
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

    STRICT_CTX_ONLY = False

    CHAT_HISTORY_SIZE = 2


if in_prod():
    config_path = "./config.toml" # default same dir as executable

    parser = ArgumentParser()
    parser.add_argument("exe") # ./main itself, just ignore
    parser.add_argument("-cfg", "--config", type=str, required=False)
    arg = parser.parse_args(sys.argv)

    if arg.config is not None:
        config_path = arg.config
else:
    config_path = "../dev.toml" # outside project folder

try:
    with Toml(config_path) as t:
        t(Config)

        db_path = t.parse("db.path")
        Config.QDRANT.ENV["QDRANT__STORAGE__STORAGE_PATH"] = db_path
        Config.QDRANT.ENV["QDRANT__STORAGE__SNAPSHOTS_PATH"] = f"{db_path}/snapshots"

except Exception as e:
    print(e)
    sys.exit()

# argv overrides config
# if in_prod():
#     Config.XYZ = arg.xyz
