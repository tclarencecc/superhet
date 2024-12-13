from llama_cpp import Llama, llama_chat_format
from typing import Iterable, Self
import time
import json
import gc

from common.decorator import suppress_print

class Llm:
    class Config:
        def __init__(self, d: dict):
            self.model = ""
            self.n_ctx = 0
            self.lora_path = None
            self.lora_scale = 1.0
            self.flash_attn = False
            self.embedding = False
            self.temperature = 0.0
            self.debug = False

            for k, v in d.items():
                setattr(self, k, v)

    def __init__(self, d: dict):
        self._config = Llm.Config(d)

        @suppress_print((
            "Model metadata:",
            "Using gguf chat template:",
            "Available chat formats",
            "Using chat eos_token:",
            "Using chat bos_token:",
            "llm_load_vocab:",
            "llama_new_context_with_model:",
            "ggml_metal_init: loaded kernel",
            "ggml_metal_init: skipping",
            "llm_load_print_meta:"))
        def ctor():
            return Llama(self._config.model,
                n_gpu_layers=-1,
                n_ctx=self._config.n_ctx,
                lora_path=self._config.lora_path,
                lora_scale=self._config.lora_scale,
                flash_attn=self._config.flash_attn,
                embedding=self._config.embedding,
                verbose=self._config.debug
            )

        self._llama = ctor()
        # on Llama constructor, verbose is initialized based on cfg.debug
        # to disable runtime verbosity, set false after constructor call i.e. llama_perf_context_print
        self._llama.verbose = False

        self._comp_text: str = None
        self._comp_chat: list = None
        self._emb_text: str = None

        self._grammar = None
        self._benchmark = False

    def close(self):
        del self._llama
        gc.collect()

    # grammar = {
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
    def __call__(self, input: str | dict, grammar: dict=None, benchmark=False) -> Self:
        """
        input dict type format:\n
        {
            system: str,
            user: str,
            chat: [tuple[str, str]]
        }
        """
        self._comp_text = None
        self._comp_chat = None
        self._grammar = grammar
        self._benchmark = benchmark

        if type(input) is str:
            if self._config.embedding:
                self._emb_text = input
            else:
                self._comp_text = input

        elif type(input) is dict:
            msg = []
            if "system" in input:
                msg.append({
                    "role": "system",
                    "content": input["system"]
                })
            if "chat" in input:
                chat: list[tuple[str, str]] = input["chat"]
                for v in chat:
                    msg.append({
                        "role": "user",
                        "content": v[0]
                    })
                    msg.append({
                        "role": "assistant",
                        "content": v[1]
                    })
            if "user" in input:
                msg.append({
                    "role": "user",
                    "content": input["user"]
                })

            self._comp_chat = msg
        
        return self
    
    def _completion(self, stream: bool) -> any:
        grammar = None
        if self._grammar is not None:
            grammar = llama_chat_format._grammar_for_response_format({
                "type": "json_object",
                "schema": self._grammar
            })

        return self._llama.create_completion(self._comp_text,
            max_tokens=None,
            stream=stream,
            temperature=self._config.temperature,
            grammar=grammar
        )
    
    def _chat_completion(self, stream: bool) -> any:
        response_format = None
        if self._grammar is not None:
            response_format = {
                "type": "json_object",
                "schema": self._grammar
            }

        return self._llama.create_chat_completion(self._comp_chat,
            max_tokens=None,
            stream=stream,
            temperature=self._config.temperature,
            response_format=response_format
        )
    
    @property
    def stats(self) -> dict:
        return {
            "n_embd": self._llama._model.n_embd(),
            "n_ctx_train": self._llama._model.n_ctx_train()
        }
    
    @property
    def stream(self) -> Iterable[str]:
        count = 0
        t = time.time()

        if self._comp_text is not None:
            for r in self._completion(True):
                count += 1
                yield r["choices"][0]["text"]
                
        elif self._comp_chat is not None:
            for r in self._chat_completion(True):
                count += 1
                if "content" in r["choices"][0]["delta"]:
                    yield r["choices"][0]["delta"]["content"]

        if self._benchmark:
            t = time.time() - t
            yield f"\n{t:.1f} sec @ {(count / t):.1f} token/sec"

    @property
    def static(self) -> str:
        if self._comp_text is not None:
            res = self._completion(False)
            return res["choices"][0]["text"]
        
        elif self._comp_chat is not None:
            res = self._chat_completion(False)
            return res["choices"][0]["message"]["content"]

    @property
    def json(self) -> any:
        return json.loads(self.static)
    
    @property
    def embed(self) -> list[float]:
        res = self._llama.create_embedding(self._emb_text)
        return res["data"][0]["embedding"]
