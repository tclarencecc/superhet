from aiohttp import ClientSession
from config import Config
from util import benchmark, HttpError

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
        async with session.post(Config.LLAMA.HOST + "/completion",
            headers={ "Authorization": "Bearer " + Config.LLAMA.KEY },
            json={ "prompt": prompt }
        ) as res:
            if res.status != 200:
                raise HttpError("llm.completion returned error status: " + str(res.status))
            
            json = await res.json()
            return json["content"]
