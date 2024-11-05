from llama_cpp import Llama, CreateEmbeddingResponse
from typing import Iterable, Callable, Iterator
import time
from collections import deque

from app.config import Config, PromptFormat
from app.util import MutableString

_NO_CTX_ANS_MSG = "Unable to answer the question as there is not enough context provided."

class Chat:
    class Entry:
        def __init__(self, query: str):
            self.query = query
            self.cot = ""
            self.answer: any = None # can be of any type depending on usage

    def __init__(self):
        self._deque: deque[Chat.Entry] = deque([], Config.CHAT_HISTORY_SIZE)
        
    @property
    def latest(self) -> Entry | None:
        last: Chat.Entry = None
        for v in self._deque:
            if v is not None:
                last = v

        return last
    
    def _prompt_formatter(self, query: str, ctx: str, cot=False, history=False) -> str:
        ctx = f"Context: {ctx}" if ctx != "" else ""

        cot = """You are an AI assistant designed to provide detailed, step-by-step responses.
First, use a [thinking] section to analyze the question and outline your approach.
Second, use a [steps] section to list how to solve the problem using a Chain of Thought reasoning process.
Third, use a [reflection] section where you review reasoning, check for errors, and confirm or adjust conclusions.
Fourth, provide the final answer in an [output] section.""" if cot else ""
        
        only = "Answer using provided context only." if ctx != "" else ""

        ret = MutableString()

        def hist(fn):
            if history:
                for v in self._deque:
                    if v is not None:
                        ret.add(fn(v.query, v.cot))

        if Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.CHATML:
            # <|im_start|>system
            # {}<|im_end|>
            # <|im_start|>user
            # {}<|im_end|>
            # <|im_start|>assistant
            def template(req, res=None):
                return f"""<|im_start|>user
{req}<|im_end|><|im_start|>assistant
{f'{res}<|im_end|>' if res is not None else ''}"""
            
            # qwen forgets system prompt if lengthy history is present
            hist(template)
            ret.add(template(f"{cot}\n{ctx}\n{query}\n{only}"))
        
        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.GEMMA:
            # <start_of_turn>user
            # {}<end_of_turn>
            # <start_of_turn>model
            def template(req, res=None):
                return f"""<start_of_turn>user
{req}<end_of_turn><start_of_turn>model
{f'{res}<end_of_turn>' if res is not None else ''}"""

            hist(template)
            # gemma has no system prompt, add to user prompt instead
            ret.add(template(f"{cot}\n{ctx}\n{query}\n{only}"))

        elif Config.LLAMA.COMPLETION.PROMPT_FORMAT == PromptFormat.LLAMA:
            # <|start_header_id|>system<|end_header_id|>{}<|eot_id|>
            # <|start_header_id|>user<|end_header_id|>{}<|eot_id|>
            # <|start_header_id|>assistant<|end_header_id|>
            def template(req, res=None):
                return f"""<|start_header_id|>user<|end_header_id|>
{req}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{f'{res}<|eot_id|>' if res is not None else ''}"""
            
            ret.add(f"<|start_header_id|>system<|end_header_id|>{cot}<|eot_id|>")
            hist(template)
            ret.add(template(f"{ctx}\n{query}\n{only}"))

        return ret.value()

    def completion_prompt(self, query: str, ctx: str) -> str | None:
        # ctx stc his
        # 1   -   -   ctx only
        # 0   1   0   no ans
        # 0   1   1   hist as ctx only
        # 0   0   -   free

        if ctx == "" and Config.STRICT_CTX_ONLY:
            if self.latest is None or self.latest.cot == "":
                self._deque.append(Chat.Entry(query))
                return
            else:
                ctx = self.latest.cot

        # build prompt first bef adding new chat entry so that it doesnt bec part of history!
        ret = self._prompt_formatter(query, ctx, cot=True, history=True)
        self._deque.append(Chat.Entry(query))

        return ret
    
    def extraction_prompt(self) -> str:
        return self._prompt_formatter("""Extract the output only, do not add anything else.
If no output can be found, summarize the provided context.""", self.latest.cot)


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
        prompt = chat.completion_prompt(query, ctx)
        if prompt is None:
            yield f"\n{_NO_CTX_ANS_MSG}"
            
        else:
            res = self._llm.create_completion(prompt,
                max_tokens=None,
                temperature=Config.LLAMA.COMPLETION.TEMPERATURE,
                stream=True
            )

            count = 0
            cot = MutableString()
            t = time.time()

            while (r := next(res, None)) is not None:
                count += 1
                v = r["choices"][0]["text"]
                cot.add(v)
                yield v

            chat.latest.cot = cot.value()
            t = time.time() - t # completion done at this point, time it

            res = self._llm.create_completion(chat.extraction_prompt(),
                max_tokens=None,
                temperature=Config.LLAMA.COMPLETION.TEMPERATURE,
                stream=False
            )

            chat.latest.answer = res["choices"][0]["text"]
            #print(f"\n{chat.latest.answer}")
            
            yield f"\n{t:.1f} sec @ {(count / t):.1f} token/sec"


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

        for _ in range(Config.LLAMA.EMBEDDING.BATCH_SIZE):
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
    