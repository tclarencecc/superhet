import asyncio

from app.util import extprocess
from app.config import Config
from app.cli import cli

@extprocess([
    (Config.QDRANT.PATH, Config.QDRANT.SHELL, Config.QDRANT.ENV),
    (Config.LLAMA.PATH, Config.LLAMA.SHELL)
    ])
def main():
    asyncio.run(cli())

main()
