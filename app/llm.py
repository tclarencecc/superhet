from llama_cpp import Llama, CreateEmbeddingResponse, CreateCompletionResponse
from typing import Iterable, Callable, Iterator
import time

from app.config import Config

# https://onnxruntime.ai/_app/immutable/assets/Phi2_Int4_TokenGenerationTP.ab4c4b44.png
# at batch size 4, llama cpp embedding approaches speed of onnxruntime (1.14x)
_BATCH_SIZE = 4
_NO_RECORD_MSG = "Unable to answer as no data can be found in the record."

def completion(ctx: str, query: str, stream: bool=False) -> str | Iterator[CreateCompletionResponse]:
    if ctx == "":
        return _NO_RECORD_MSG

    prompt = """<|im_start|>system
You are a helpful assistant. Answer using provided context only. Context: {ctx}
<|im_end|>
<|im_start|>user
{query} Answer using provided context only.
<|im_end|>
<|im_start|>assistant
""".format(ctx=ctx, query=query)
        
    llm = Llama(Config.LLAMA.COMPLETION.MODEL,
        n_gpu_layers=-1,
        n_ctx=0,
        flash_attn=Config.LLAMA.COMPLETION.FLASH_ATTENTION,
        verbose=False
    )
    res = llm.create_completion(prompt,
        max_tokens=None,
        temperature=Config.LLAMA.COMPLETION.TEMPERATURE,
        stream=stream
    )

    if stream:
        return res
    else:
        return res["choices"][0]["text"]
    
def completion_stream(ctx: str, query: str) -> Iterator[str]:
    if ctx == "":
        yield _NO_RECORD_MSG
    else:
        count = 0
        t = time.time()
        res = completion(ctx, query, stream=True)

        while (r := next(res, None)) is not None:
            count += 1
            yield r["choices"][0]["text"]

        t = time.time() - t
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
    