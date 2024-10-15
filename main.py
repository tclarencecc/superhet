import asyncio
from httpx import AsyncClient

from app.config import Config
from app.cli import cli
from app.db import Db
import app.llm as llm

async def app():
    def post_config_load():
        # setup non-toml based config values
        n_embd, n_ctx = llm.Embedding.stats()
        Config.LLAMA.EMBEDDING.SIZE = n_embd
        Config.LLAMA.EMBEDDING.CONTEXT = n_ctx

    Config.load_from_toml(post_config_load)

    async with AsyncClient() as client, Db(client):
        await cli()

try:
    asyncio.run(app())
except KeyboardInterrupt:
    pass
