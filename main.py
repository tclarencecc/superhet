import asyncio
from httpx import AsyncClient

from app.util import extprocess
from app.config import Config
from app.cli import cli
import app.db as db
import app.llm as llm

async def app():
    async with AsyncClient() as client:
        # set up all http reliant services
        db.http_client(client)

        # set up runtime, non-argv config values
        n_embd, n_ctx = llm.Embedding.stats()
        Config.LLAMA.EMBEDDING.SIZE = n_embd
        Config.LLAMA.EMBEDDING.CONTEXT = n_ctx

        await cli()

@extprocess([
    (Config.QDRANT.PATH, Config.QDRANT.SHELL, Config.QDRANT.ENV)
    ])
def main():
    asyncio.run(app())

main()
