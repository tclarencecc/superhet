from aiohttp import ClientSession

from app.config import Config
from app.util import benchmark

class LlmError(Exception): ...

@benchmark("llm completion")
async def completion(ctx: str, query: str) -> str:
    if ctx == "":
        return "Unable to answer as no data can be found in the record."

    # de-indent to save whitespace in ctx window text
    prompt = """<|im_start|>system
You are a helpful assistant. Answer using provided context only.
Context: {ctx}
<|im_end|>
<|im_start|>user
{query} Answer using provided context only.
<|im_end|>
<|im_start|>assistant
""".format(ctx=ctx, query=query)

    async with ClientSession() as session:
        async with session.post(f"{Config.LLAMA.HOST}/completion",
            headers={ "Authorization": f"Bearer {Config.LLAMA.KEY}" },
            json={ "prompt": prompt }
        ) as res:
            if res.status != 200:
                raise LlmError(f"llm.completion returned error status: {res.status}")
            
            json = await res.json()
            return json["content"]

async def ready() -> bool:
    async with ClientSession() as session:
        async with session.get(f"{Config.LLAMA.HOST}/health") as res:
            if res.status == 503: # the other 503 is from fail_on_no_slot which is not used here
                return False
            elif res.status == 500:
                raise LlmError("llm.ready returned model failed to load error.")
            else: # only 200 remains
                return True
