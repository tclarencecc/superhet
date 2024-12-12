from llama_cpp import Llama
from typing import Iterable, Iterator
import time
from collections import deque

from agent.config import Config, PromptFormat
from common.string import MutableString
from agent.llm_base import Llm, _Llama

class Chat:
    class Entry:
        def __init__(self):
            self.query = ""
            self.context = ""
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

    def to_list(self) -> list[tuple[str, str]]:
        ret = []
        for v in self._deque:
            if v is not None:
                ret.append((v.query, v.answer))
        return ret


class Completion:
    _llm: _Llama = None

    @staticmethod
    def init():
        if Completion._llm is None:
            Llm["Completion"] = {
                "model": Config.LLAMA.COMPLETION.MODEL,
                "n_ctx": Config.LLAMA.COMPLETION.CONTEXT_SIZE,
                "flash_attn": Config.LLAMA.COMPLETION.FLASH_ATTENTION,
                "debug": Config.DEBUG
            }
            Completion._llm = Llm["Completion"]

    @staticmethod
    def run(query: str, ctx: str, chat: Chat) -> Iterator[str]:
        Completion.init()

        #  w/ctx? - N, strict? - Y, query? - Y, no ans  (1)
        #                                    N, ~chat   (2)
        #                        N, query? - Y, cot     (1)
        #                                    N, ~chat   (2)
        #         - Y, ctx cot                          (3)
        # ~: implicitly use
        # testing seq: 1-1-2-3-2-1-2

        cot = """First, use a [thinking] section to analyze the question and outline your approach.
Second, use a [steps] section to list how to solve the problem using a Chain of Thought reasoning process.
Third, use a [reflection] section where you review reasoning, check for errors, and confirm or adjust conclusions.
Fourth, provide the final answer in an [output] section."""

        if ctx == "":
            if Config.STRICT_CTX_ONLY:
                prompt = {
                    "system": "If question requires analysis, do not answer and say 'Not enough context to answer'.",
                    "user": query,
                    "chat": chat.to_list()
                }
            else:
                prompt = {
                    "system": f"If question requires analysis, {cot}",
                    "user": query,
                    "chat": chat.to_list()
                }
        else:
            prompt = {
                "system": cot,
                "user": f"Context: {ctx}\n{query}\nAnswer using context only.",
                "chat": []
            }

        # GEMMA have no system prompt!
        if Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.GEMMA:
            prompt["user"] = f"{prompt["system"]}\n{prompt["user"]}"
            del prompt["system"]

        count = 0
        cot = MutableString()
        t = time.time()

        for r in Completion._llm(prompt).stream:
            count += 1
            cot.add(r)
            yield r

        t = time.time() - t
        yield f"\n{t:.1f} sec @ {(count / t):.1f} token/sec"

        entry = Chat.Entry()
        entry.query = query
        entry.context = ctx
        entry.answer = Completion._llm(f"""Context: {cot.value()}
Extract the output only, do not add anything. If no output can be found, summarize the context.""").static
        chat.add(entry)
        
        
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
    