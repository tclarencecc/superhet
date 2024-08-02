import asyncio

from app.util import extprocess
from app.config import Config
from app.cli import cli

@extprocess([
    (Config.QDRANT.PATH, Config.QDRANT.SHELL, Config.QDRANT.ENV)
    ])
def main():
    asyncio.run(cli())

main()
