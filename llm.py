import subprocess
import shlex
import requests

def _en_prompt() -> str:
    return """
    <|im_start|>system
    You are a helpful assistant. Answer using provided context only.
    Context: {ctx}
    <|im_end|>
    <|im_start|>user
    {query} Answer using provided context only.
    <|im_end|>
    <|im_start|>assistant
    """

class Llm:
    def __init__(self, host="", cli_path="", model_path="") -> None:
        if cli_path != "" and model_path != "":
            self.server_mode = False
            self.cli_path = cli_path
            self.model_path = model_path
        else:
            if host == "":
                raise Exception("Llm not initialized as cli or server mode.")
            self.server_mode = True
            self.host = host

    def run(self, ctx: str, query: str) -> str | None:
        prompt = _en_prompt().format(ctx=ctx, query=query)

        if self.server_mode:
            if ctx == "":
                return "Unable to answer as no data can be found in the record."

            res = requests.post(self.host + "/completion",
                json={ "prompt": prompt }
            )
            if res.status_code != 200:
                raise Exception("llm server returned error status: " + str(res.status_code))

            return res.json()["content"]
        else:
            if ctx == "":
                print("Unable to answer as no data can be found in the record.")
                return

            prompt = shlex.quote(prompt) # escape to prevent code injection

            cmd = "{cli} -m {model} -fa -p {prompt}"
            cmd = cmd.format(cli=self.cli_path, model=self.model_path, prompt=prompt)

            subprocess.run(shlex.split(cmd))
