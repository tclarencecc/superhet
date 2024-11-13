from llama_cpp import Llama
from typing import Iterable, Iterator
import time
from collections import deque

from app.config import Config, PromptFormat
from common.string import MutableString
from common.decorator import suppress_print

class Chat:
    class Entry:
        def __init__(self):
            self.query = ""
            self.context = ""
            self.cot = ""
            self.answer = ""

    def __init__(self):
        self._deque: deque[Chat.Entry] = deque([], Config.CHAT_HISTORY_SIZE)
        
    @property
    def latest(self) -> Entry | None:
        last: Chat.Entry = None
        for v in self._deque:
            if v is not None:
                last = v
        return last

    def add(self, entry: Entry):
        self._deque.append(entry)


class Completion:
    class _Executor:
        def __init__(self, prompt: str):
            self._prompt = prompt

        @property
        def stream(self) -> Iterable[str]:
            res = Completion._llm().create_completion(self._prompt,
                max_tokens=None,
                temperature=Config.LLAMA.COMPLETION.TEMPERATURE,
                stream=True
            )
            for r in res:
                yield r["choices"][0]["text"]

        @property
        def static(self) -> str:
            res = Completion._llm().create_completion(self._prompt,
                max_tokens=None,
                temperature=Config.LLAMA.COMPLETION.TEMPERATURE
            )
            return res["choices"][0]["text"]

    _instance = None

    @staticmethod
    def _llm() -> Llama:
        if Completion._instance is None:
            @suppress_print(("Model metadata:", "Using gguf chat template:", "Available chat formats",
                "Using chat eos_token:", "Using chat bos_token:"))
            def ctor():
                return Llama(Config.LLAMA.COMPLETION.MODEL,
                    n_gpu_layers=-1,
                    n_ctx=Config.LLAMA.COMPLETION.CONTEXT_SIZE,
                    flash_attn=Config.LLAMA.COMPLETION.FLASH_ATTENTION,
                    verbose=Config.DEBUG
                )
            Completion._instance = ctor()
            # disable llama_perf_context_print, which logs after each create_completion
            Completion._instance.verbose = False

        return Completion._instance

    @staticmethod
    def _exec(user: str, system="", chat: Chat=None) -> _Executor:
        prompt = MutableString()

        def hist(fn):
            if chat is not None:
                for v in chat._deque:
                    if v is not None:
                        prompt.add(fn(v.query, v.answer))

        if Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.CHATML:
            # <|im_start|>system
            # {}<|im_end|>
            # <|im_start|>user
            # {}<|im_end|>
            # <|im_start|>assistant
            def template(usr, asst=None):
                return f"""<|im_start|>user
{usr}<|im_end|><|im_start|>assistant
{f'{asst}<|im_end|>' if asst is not None else ''}"""
            
            prompt.add(f"<|im_start|>system\n{system}<|im_end|>")
            hist(template)
            prompt.add(template(user))

        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.GEMMA:
            # <start_of_turn>user
            # {}<end_of_turn>
            # <start_of_turn>model
            def template(usr, asst=None):
                return f"""<start_of_turn>user
{usr}<end_of_turn><start_of_turn>model
{f'{asst}<end_of_turn>' if asst is not None else ''}"""

            hist(template)
            # gemma has no system prompt, add to user prompt instead
            prompt.add(template(f"{system}\n{user}"))

        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.LLAMA:
            # <|start_header_id|>system<|end_header_id|>{}<|eot_id|>
            # <|start_header_id|>user<|end_header_id|>{}<|eot_id|>
            # <|start_header_id|>assistant<|end_header_id|>
            def template(usr, asst=None):
                return f"""<|start_header_id|>user<|end_header_id|>
{usr}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{f'{asst}<|eot_id|>' if asst is not None else ''}"""
            
            prompt.add(f"<|start_header_id|>system<|end_header_id|>{system}<|eot_id|>")
            hist(template)
            prompt.add(template(user))

        return Completion._Executor(prompt.value())
        
    @staticmethod
    def run(query: str, ctx: str, chat: Chat) -> Iterator[str]:
        #  w/ctx? - N, strict? - Y, query? - Y, no ans  (1)
        #                                    N, ~chat   (2)
        #                        N, query? - Y, cot     (1)
        #                                    N, ~chat   (2)
        #         - Y, ctx cot                          (3)
        # ~: implicitly use
        # testing seq: 1-1-2-3-2-1-2

        #role = "You are an AI assistant designed to provide detailed, step-by-step responses."
        cot = """First, use a [thinking] section to analyze the question and outline your approach.
Second, use a [steps] section to list how to solve the problem using a Chain of Thought reasoning process.
Third, use a [reflection] section where you review reasoning, check for errors, and confirm or adjust conclusions.
Fourth, provide the final answer in an [output] section."""
      
        if ctx == "":
            if Config.STRICT_CTX_ONLY:
                ex = Completion._exec(query, chat=chat,
system="If question requires analysis, do not answer and say 'Not enough context to answer'.")
            else:
                ex = Completion._exec(query, chat=chat, system=f"If question requires analysis, {cot}")
        else:
            ex = Completion._exec(f"Context: {ctx}\n{query}\nAnswer using context only.", system=cot)

        count = 0
        cot = MutableString()
        t = time.time()

        for r in ex.stream:
            count += 1
            cot.add(r)
            yield r

        t = time.time() - t

        entry = Chat.Entry()
        entry.query = query
        entry.context = ctx
        entry.cot = cot.value()
        entry.answer = Completion._exec(f"""Context: {entry.cot}
Extract the output only, do not add anything. If no output can be found, summarize the context.""").static
        chat.add(entry)
        #print(f"\n{entry.answer}")
        
        yield f"\n{t:.1f} sec @ {(count / t):.1f} token/sec"


class Embedding:
    _instance = None

    @staticmethod
    def _llm() -> Llama:
        if Embedding._instance is None:
            Embedding._instance = Llama(Config.LLAMA.EMBEDDING.MODEL,
                n_gpu_layers=-1,
                n_ctx=0,
                embedding=True,
                verbose=False
            )
        return Embedding._instance
        
    @staticmethod
    def stats() -> tuple[int, int]:
        """
        stats of the embedding model\n
        returns: [embedding dimension, context length]
        """
        return (Embedding._llm()._model.n_embd(), Embedding._llm()._model.n_ctx_train())

    @staticmethod
    def from_string(input: str) -> list[float]:
        """
        adhoc convert single string to vector
        """
        res = Embedding._llm().create_embedding(input)
        return res["data"][0]["embedding"]

    def __init__(self, input: Iterable[str]):
        self._iterable = input
        self._eof = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._eof:
            raise StopIteration
        
        chunks = []
        vectors = []

        for _ in range(Config.LLAMA.EMBEDDING.BATCH_SIZE):
            chunk = next(self._iterable, None)
            if chunk is None:
                self._eof = True
                break

            res = Embedding._llm().create_embedding(chunk)
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
    