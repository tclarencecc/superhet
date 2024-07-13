import asyncio
from util import extprocess
from config import Config
from cli import cli

@extprocess([
    (Config.QDRANT.PATH, Config.QDRANT.SHELL, Config.QDRANT.ENV),
    (Config.LLAMA.PATH, Config.LLAMA.SHELL)
    ])
def main():
    asyncio.run(cli())

main()
