from llama_cpp import Llama, CreateEmbeddingResponse
from typing import Iterable, Callable, Iterator
import time
from collections import deque

from app.config import Config, PromptFormat
from app.util import MutableString

# https://onnxruntime.ai/_app/immutable/assets/Phi2_Int4_TokenGenerationTP.ab4c4b44.png
# at batch size 4, llama cpp embedding approaches speed of onnxruntime (1.14x)
_BATCH_SIZE = 4
_NO_CTX_ANS_MSG = "Unable to answer the question as there is not enough context provided."

class Chat:
    class Entry:
        def __init__(self, req: str):
            self.req = req
            self.res = ""
            self.ctx_ans = True

    def __init__(self):
        self._deque: deque[Chat.Entry] = deque([], Config.CHAT_HISTORY_SIZE)

    @property
    def output_tag(self) -> str:
        if Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.CHATML:
            return "[output]"
        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.GEMMA:
            return "[output]"
        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.LLAMA:
            return "**output**"
        
    @property
    def latest(self) -> Entry | None:
        last: Chat.Entry = None
        for v in self._deque:
            if v is not None:
                last = v

        return last

    def request(self, query: str, ctx: str) -> str | None:
        # ctx his scenario
        # 0   0   1st query
        #         1) strict: no ans
        #         2) not:    free
        # 0   1   nth query, follow up
        #         strict:
        #           3) prev no:        no ans
        #           4) prev with:      ctx only = prev res
        #         5) not:    free
        # 1   0   1st query
        # 1   1   nth, follow up
        #         6) all:    ctx only

        if ctx == "" and Config.STRICT_CTX_ONLY:
            if self.latest is None:
                self._deque.append(Chat.Entry(query))
                return # 1
            elif not self.latest.ctx_ans:
                self._deque.append(Chat.Entry(query))
                return # 3
            else:
                ctx = f"Context: {self.latest.res}" # 4
        else:
            ctx = f"Context: {ctx}" if ctx != "" else "" # 2,5 = "", 6 = ctx

        only = " Answer using provided context only." if ctx != "" else "" # 4,6
        
        cot = """You are an AI assistant designed to provide detailed, step-by-step responses.
First, use a [thinking] section to analyze the question and outline your approach.
Second, use a [steps] section to list how to solve the problem using a Chain of Thought reasoning process.
Third, use a [reflection] section where you review reasoning, check for errors, and confirm or adjust conclusions.
Fourth, provide the final answer in an [output] section."""

        # use {req} {res} as placeholders
        def reqres(template: str, input: MutableString):
            for v in self._deque:
                if v is not None:
                    input.add(template.format(req=v.req, res=v.res))

        prompt = MutableString()

        if Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.CHATML:
            # <|im_start|>system
            # {}
            # <|im_end|>
            # <|im_start|>user
            # {}
            # <|im_end|>
            # <|im_start|>assistant

            # qwen often forgets system prompt cot in long conversations, add it to user prompt instead
            #prompt.add(f"<|im_start|>system\n{cot}<|im_end|>")
            reqres("<|im_start|>user\n{req}<|im_end|><|im_start|>assistant\n{res}<|im_end|>",
                prompt)
            prompt.add(f"<|im_start|>user\n{cot}\n{ctx}\n{query}{only}<|im_end|><|im_start|>assistant")
        
        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.GEMMA:
            # <start_of_turn>user
            # {}
            # {}<end_of_turn>
            # <start_of_turn>model

            reqres("<start_of_turn>user\n{req}<end_of_turn><start_of_turn>model\n{res}<end_of_turn>",
                prompt)
            prompt.add(f"<start_of_turn>user\n{cot}\n{ctx}\n{query}{only}<end_of_turn><start_of_turn>model")

        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.LLAMA:
            # <|start_header_id|>system<|end_header_id|>{}<|eot_id|>
            # <|start_header_id|>user<|end_header_id|>{}<|eot_id|>
            # <|start_header_id|>assistant<|end_header_id|>

            prompt.add(f"<|start_header_id|>system<|end_header_id|>{cot}<|eot_id|>")
            reqres("<|start_header_id|>user<|end_header_id|>{req}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{res}<|eot_id|>",
                prompt)
            prompt.add(f"<|start_header_id|>user<|end_header_id|>{ctx}\n{query}{only}<|eot_id|><|start_header_id|>assistant<|end_header_id|>")

        self._deque.append(Chat.Entry(query))

        return prompt.value()

    def response(self, value: str):
        last: Chat.Entry = None
        for v in self._deque:
            if v is not None:
                last = v

        if last is not None:
            last.res = value
            if value == _NO_CTX_ANS_MSG:
                last.ctx_ans = False


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

    def __call__(self, query: str, ctx: str, chat: Chat) -> Iterator[str]:
        prompt = chat.request(query, ctx)
        if prompt is None:
            chat.response(_NO_CTX_ANS_MSG)
            yield f"\n{_NO_CTX_ANS_MSG}"

        else:
            res = self._llm.create_completion(prompt,
                max_tokens=None,
                temperature=Config.LLAMA.COMPLETION.TEMPERATURE,
                stream=True
            )

            count = 0
            comp = MutableString()
            t = time.time()

            while (r := next(res, None)) is not None:
                count += 1
                v = r["choices"][0]["text"]
                comp.add(v)
                yield v

            t = time.time() - t
            yield f"\n{t:.1f} sec @ {(count / t):.1f} token/sec"

            # comp: xxx<output tag>yyy
            raw = comp.value().lower()

            spl_raw = raw.split(chat.output_tag)
            if len(spl_raw) == 2:
                chat.response(spl_raw[1].strip())


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
    