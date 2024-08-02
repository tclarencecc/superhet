from llama_cpp import Llama

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

    llm = Llama(Config.LLAMA.MODEL,
        n_gpu_layers=1,
        n_ctx=8000,
        flash_attn=True,
        verbose=False
    )
    res = llm.create_completion(prompt,
        max_tokens=None,
        temperature=Config.LLAMA.TEMPERATURE
    )

    return res["choices"][0]["text"]
