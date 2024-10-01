from llama_cpp import Llama, CreateEmbeddingResponse
from typing import Iterable, Callable, Iterator
import time
from collections import deque

from app.config import Config, PromptFormat
from app.util import MutableString

# https://onnxruntime.ai/_app/immutable/assets/Phi2_Int4_TokenGenerationTP.ab4c4b44.png
# at batch size 4, llama cpp embedding approaches speed of onnxruntime (1.14x)
_BATCH_SIZE = 4
_NO_RECORD_MSG = "Unable to answer as no data can be found in the record."

class Chat:
    class Entry:
        def __init__(self, req: str, res: str):
            self.req = req
            self.res = res

    @staticmethod
    def prompt(query: str, ctx: str, history: Iterable[Entry]=None) -> str:
        ctx = f"Context: {ctx}" if ctx != "" else ""
        only = " Answer using provided context only." if ctx != "" else ""

        cot = """You are an AI assistant designed to provide detailed, step-by-step responses.
First, use a [thinking] section to analyze the question and outline your approach.
Second, use a [steps] section to list how to solve the problem using a Chain of Thought reasoning process.
Third, use a [reflection] section where you review reasoning, check for errors, and confirm or adjust conclusions.
Fourth, provide the final answer in an [output] section."""

        # use {req} {res} as placeholders
        def reqres(template: str) -> str:
            ret = ""
            if history is not None:
                for v in history:
                    if v is not None:
                        ret += template.format(req=v.req, res=v.res)
            return ret

        if Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.CHATML:
            # <|im_start|>system
            # {}
            # <|im_end|>
            # <|im_start|>user
            # {}
            # <|im_end|>
            # <|im_start|>assistant
            prompt = f"<|im_start|>system\n{cot}<|im_end|>"
            prompt += reqres("<|im_start|>user\n{req}<|im_end|><|im_start|>assistant\n{res}<|im_end|>")
            prompt += f"<|im_start|>user\n{ctx}\n{query}{only}<|im_end|><|im_start|>assistant"
        
        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.GEMMA:
            # <start_of_turn>user
            # {}
            # {}<end_of_turn>
            # <start_of_turn>model
            prompt = reqres("<start_of_turn>user\n{req}<end_of_turn><start_of_turn>model\n{res}<end_of_turn>")
            prompt += f"<start_of_turn>user\n{cot}\n{ctx}\n{query}{only}<end_of_turn><start_of_turn>model"

        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.LLAMA:
            # <|start_header_id|>system<|end_header_id|>{}<|eot_id|>
            # <|start_header_id|>user<|end_header_id|>{}<|eot_id|>
            # <|start_header_id|>assistant<|end_header_id|>
            prompt = f"<|start_header_id|>system<|end_header_id|>{cot}<|eot_id|>"
            prompt += reqres(
"""<|start_header_id|>user<|end_header_id|>{req}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{res}<|eot_id|>""")
            prompt += f"<|start_header_id|>user<|end_header_id|>{ctx}\n{query}{only}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        return prompt
    
    @staticmethod
    def output_tag() -> str:
        if Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.CHATML:
            return "[output]"
        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.GEMMA:
            return "[output]"
        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.LLAMA:
            return "**output**"

    def __init__(self):
        self._deque: deque[Chat.Entry] = deque([], Config.CHAT_HISTORY_SIZE)

    def request(self, query: str, ctx: str) -> str:
        ret = Chat.prompt(query, ctx, self._deque)
        self._deque.append(Chat.Entry(query, ""))
        return ret

    def response(self, value: str):
        last: Chat.Entry = None
        for v in self._deque:
            if v is not None:
                last = v

        if last is not None:
            last.res = value
        
class Completion:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Completion, cls).__new__(cls)

        return cls._instance

    def __init__(self):
        self._llm = Llama(Config.LLAMA.COMPLETION.MODEL,
            n_gpu_layers=-1,
            n_ctx=Config.LLAMA.COMPLETION.CONTEXT_SIZE,
            flash_attn=Config.LLAMA.COMPLETION.FLASH_ATTENTION,
            verbose=False
        )

    def __call__(self, query: str, ctx: str, chat: Chat=None) -> Iterator[str]:
        chat_mode = chat is not None

        if chat_mode:
            prompt = chat.request(query, ctx)
            comp = MutableString()
        else:
            prompt = Chat.prompt(query, ctx)

        res = self._llm.create_completion(prompt,
            max_tokens=None,
            temperature=Config.LLAMA.COMPLETION.TEMPERATURE,
            stream=True
        )

        count = 0
        t = time.time()

        while (r := next(res, None)) is not None:
            count += 1
            v = r["choices"][0]["text"]
            if chat_mode:
                comp.add(v)
            yield v

        t = time.time() - t
        yield f"\n{t:.1f} sec @ {(count / t):.1f} token/sec"

        if chat_mode:
            # comp: xxx<output tag>yyy
            raw = comp.value().lower()

            spl_raw = raw.split(Chat.output_tag())
            if len(spl_raw) == 2:
                chat.response(spl_raw[1].strip())
            # else:
            #     # no output tag, get the last sentence as output
            #     spl_raw = raw.split("\n")
            #     chat.response(spl_raw[-1].strip())

class Embedding:
    @staticmethod
    def _llm() -> Llama:
        return Llama(Config.LLAMA.EMBEDDING.MODEL,
            n_gpu_layers=-1,
            n_ctx=0,
            embedding=True,
            verbose=False
        )
    
    @staticmethod
    def _creator() -> Callable[[str | list[str], str | None], CreateEmbeddingResponse]:
        return Embedding._llm().create_embedding
    
    @staticmethod
    def stats() -> tuple[int, int]:
        """
        returns: [embedding dimension, context length]
        """
        llm = Embedding._llm()
        return (llm._model.n_embd(), llm._model.n_ctx_train())

    @staticmethod
    def create(input: str) -> list[float]:
        create_embedding = Embedding._creator()
        res = create_embedding(input)
        return res["data"][0]["embedding"]

    def __init__(self, input: Iterable[str]):
        self._iterable = input
        self._create_embedding = Embedding._creator()
        self._eof = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._eof:
            raise StopIteration
        
        chunks = []
        vectors = []

        for _ in range(_BATCH_SIZE):
            chunk = next(self._iterable, None)
            if chunk is None:
                self._eof = True
                break

            res = self._create_embedding(chunk)
            chunks.append(chunk)
            vectors.append(res["data"][0]["embedding"])

        # handle case where current "next" just reached the end and had nothing to return
        if len(vectors) == 0 and self._eof:
            raise StopIteration

        return {
            "documents": chunks,
            "vectors": vectors,
            "len": len(vectors)
        }
    