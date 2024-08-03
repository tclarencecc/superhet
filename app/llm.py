from llama_cpp import Llama
from typing import Iterable

from app.config import Config
from app.util import benchmark

@benchmark("llm completion")
def completion(ctx: str, query: str) -> str:
    if ctx == "":
        return "Unable to answer as no data can be found in the record."

    prompt = """<|im_start|>system
You are a helpful assistant. Answer using provided context only.
Context: {ctx}
<|im_end|>
<|im_start|>user
{query} Answer using provided context only.
<|im_end|>
<|im_start|>assistant
""".format(ctx=ctx, query=query)

    llm = Llama(Config.LLAMA.COMPLETION.MODEL,
        n_gpu_layers=1,
        n_ctx=0,
        flash_attn=Config.LLAMA.COMPLETION.FLASH_ATTENTION,
        verbose=False
    )
    res = llm.create_completion(prompt,
        max_tokens=None,
        temperature=Config.LLAMA.COMPLETION.TEMPERATURE
    )

    return res["choices"][0]["text"]

@benchmark("llm embedding")
def embedding(input: str | Iterable[str]) -> list[float] | tuple[list[str], list[list[float]]]:
    llm = Llama(Config.LLAMA.EMBEDDING.MODEL,
        n_gpu_layers=1,
        n_ctx=0,
        embedding=True,
        verbose=False
    )

    if type(input) == str:
        res = llm.create_embedding(input)
        return res["data"][0]["embedding"]
    else:
        chunks = []
        vectors = []

        while (chunk := next(input, None)) is not None:
            res = llm.create_embedding(chunk)
            chunks.append(chunk)
            vectors.append(res["data"][0]["embedding"])

        return (chunks, vectors)
