import requests
import config
from config import ConfigKey

_host = config.get(ConfigKey.LLM_HOST)

def inference(ctx: str, query: str) -> str:
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
    
    res = requests.post(_host + "/completion",
        json={ "prompt": prompt }
    )
    if res.status_code != 200:
        raise Exception("llm returned error status: " + str(res.status_code))

    return res.json()["content"]
