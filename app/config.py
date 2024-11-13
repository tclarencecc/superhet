import uuid
import sys
from argparse import ArgumentParser
from enum import Enum
from typing import Callable

from common.toml import Toml

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

class DocumentScript(Enum):
    # https://en.wikipedia.org/wiki/List_of_writing_systems
    LATIN = 1,
    HANZI = 2
    
class Config:
    class _qdrant:
        HOST = "http://localhost:6333"
        PATH = _select("./bin", "./_internal")
        SHELL = "./qdrant"
        KEY = _qdrant_key
        READ_LIMIT = 1
        # qdrant recommends multitenancy as opposed to multicollection
        COLLECTION = "my collection"
        # https://qdrant.tech/documentation/guides/configuration/#environment-variables
        ENV = {
            "QDRANT__SERVICE__API_KEY": _qdrant_key,
            "QDRANT__TELEMETRY_DISABLED": "true",
            "QDRANT__STORAGE__STORAGE_PATH": None, # manually parsed from toml
            "QDRANT__STORAGE__SNAPSHOTS_PATH": None # derived from STORAGE_PATH
        }
    QDRANT = _qdrant

    class _storage:
        SQL = Toml.Spec("storage.data", "./data")
        INDEX = Toml.Spec("storage.index", "./index")
        
        class _hnsw:
            # hardcoded
            RESIZE_STEP = 5
            MIN_DISTANCE = 0.4
            # https://qdrant.tech/documentation/guides/configuration/
            M = 16
            EF_CONSTRUCTION = 100
            #https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
            K = 1
            EF_SEARCH = 3
        HNSW = _hnsw
    STORAGE = _storage

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

            # hardcoded
            # https://onnxruntime.ai/_app/immutable/assets/Phi2_Int4_TokenGenerationTP.ab4c4b44.png
            # optimal speed at batch size 4, llama cpp vs onnxruntime (1.14x)
            BATCH_SIZE = 4
        EMBEDDING = _embedding
    LLAMA = _llama

    class _chunk:
        SEPARATOR = Toml.Spec("document.separator")
        SCRIPT = Toml.Spec("document.script", None, lambda x: DocumentScript[x])

        SIZE_LIMIT = _min_max(20, None) # no MAX; computed from EMBEDDING.CONTEXT
        SIZE = Toml.Spec("document.chunk.size")

        # https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167
        # table 5 25% overlap optimal
        OVERLAP_LIMIT = _min_max(0.1, 0.25)
        OVERLAP = Toml.Spec("document.chunk.overlap")
    CHUNK = _chunk

    BENCHMARK = not in_prod()

    PROCESS_STDOUT = False

    CLI_CMD_PREFIX = "!"

    STRICT_CTX_ONLY = False

    CHAT_HISTORY_SIZE = 5

    DEBUG = not in_prod()


    @staticmethod
    def load_from_toml(post_load_callback: Callable):
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
                t.load_to(Config)
                # setup here manually parsed config values

            post_load_callback()

            # setup post_load_callback reliant config values
            if Config.CHUNK.SCRIPT == DocumentScript.LATIN:
                # assuming a generous 2 token-per-word
                Config.CHUNK.SIZE_LIMIT.MAX = int(Config.LLAMA.EMBEDDING.CONTEXT / 2)
            elif Config.CHUNK.SCRIPT == DocumentScript.HANZI:
                # multiple chars can be just 1 token; assume worst case 1 token-per-char with small allowance
                Config.CHUNK.SIZE_LIMIT.MAX = int(Config.LLAMA.EMBEDDING.CONTEXT * 0.8)

            # validations
            def minmax_validate(val, limit: _min_max, text: str):
                if val < limit.MIN or val > limit.MAX:
                    raise ValueError(f"Config {text} must be {limit.MIN} to {limit.MAX}.")

            minmax_validate(Config.CHUNK.OVERLAP, Config.CHUNK.OVERLAP_LIMIT, "[document.chunk] overlap")
            minmax_validate(Config.CHUNK.SIZE, Config.CHUNK.SIZE_LIMIT, "[document.chunk] size")
            
        except Exception as e:
            print(e)
            sys.exit()
