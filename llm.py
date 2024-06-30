import requests

class Llm:
    def __init__(self, host: str) -> None:
        if host == "":
            raise Exception("Llm has undefined host.")
        self.host = host

    def infer(self, ctx: str, query: str) -> str:
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
"""
        res = requests.post(self.host + "/completion",
            json={ "prompt": prompt.format(ctx=ctx, query=query) }
        )
        if res.status_code != 200:
            raise Exception("llm server returned error status: " + str(res.status_code))

        return res.json()["content"]
