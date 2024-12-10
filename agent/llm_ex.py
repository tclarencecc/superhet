from llama_cpp import Llama, llama_chat_format
from typing import Iterable
import time
import json

from common.decorator import suppress_print

class _Llama:
    class _Config:
        def __init__(self, d: dict):
            self.model = ""
            self.n_ctx = 0
            self.lora_path = None
            self.lora_scale = 1.0
            self.flash_attn = False
            self.temperature = 0.0
            self.debug = True

            for k, v in d.items():
                setattr(self, k, v)

    def __init__(self, d: dict):
        self._config = _Llama._Config(d)

        @suppress_print((
            "Model metadata:",
            "Using gguf chat template:",
            "Available chat formats",
            "Using chat eos_token:",
            "Using chat bos_token:",
            "llm_load_vocab:",
            "llama_new_context_with_model:",
            "ggml_metal_init: loaded kernel",
            "ggml_metal_init: skipping"))
        def ctor():
            return Llama(self._config.model,
                n_gpu_layers=-1,
                n_ctx=self._config.n_ctx,
                lora_path=self._config.lora_path,
                lora_scale=self._config.lora_scale,
                flash_attn=self._config.flash_attn,
                verbose=self._config.debug
            )

        self._llama = ctor()
        # on Llama constructor, verbose is initialized based on cfg.debug
        # to disable runtime verbosity, set false after constructor call i.e. llama_perf_context_print
        self._llama.verbose = False
        
    def stream(self, prompt: str, timed=True) -> Iterable[str]:
        res = self._llama.create_completion(prompt,
            max_tokens=None,
            stream=True,
            temperature=self._config.temperature
        )
        count = 0
        t = time.time()

        for r in res:
            count += 1
            yield r["choices"][0]["text"]

        if timed:
            t = time.time() - t
            yield f"\n{t:.1f} sec @ {(count / t):.1f} token/sec"

    def static(self, prompt: str) -> str:
        res = self._llama.create_completion(prompt,
            max_tokens=None,
            temperature=self._config.temperature
        )
        return res["choices"][0]["text"]
    
    def json(self, prompt: str, schema: dict) -> any:
        # schema = {
        #     "type": "array",
        #     "items": {
        #         "type": "object",
        #         "properties": {
        #             "question": { "type": "string" },
        #             "answer": { "type": "string" }
        #         },
        #         "required": ["question", "answer"]
        #     },
        #     "minItems": 3,
        #     "maxItems": 100
        # }
        grammar = llama_chat_format._grammar_for_response_format({
            "type": "json_object",
            "schema": schema
        })
        res = self._llama.create_completion(prompt,
            max_tokens=None,
            temperature=self._config.temperature,
            grammar=grammar
        )
        return json.loads(res["choices"][0]["text"])


class _LlmMeta(type):
    def __getitem__(cls, name: str) -> _Llama:
        return Llm._dict()[name]
    
    def __setitem__(cls, name: str, d: dict):
        Llm._dict()[name] = _Llama(d)

class Llm(object, metaclass=_LlmMeta):
    _instance = None

    @staticmethod
    def _dict() -> dict:
        if Llm._instance is None:
            Llm._instance = {}
        return Llm._instance
