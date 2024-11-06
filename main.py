import asyncio

from app.config import Config
from app.cli import cli
from app.storage import Sql
import app.llm as llm
from app.llm import Completion, Embedding

async def app():
    def post_config_load():
        # setup non-toml based config values
        n_embd, n_ctx = llm.Embedding.stats()
        Config.LLAMA.EMBEDDING.SIZE = n_embd
        Config.LLAMA.EMBEDDING.CONTEXT = n_ctx

    Config.load_from_toml(post_config_load)

    # init static instances
    _ = Completion._llm()
    _ = Embedding._llm()

    async with Sql():
        loop = asyncio.get_event_loop()
        loop.run_until_complete(await cli())
        # run more inf loops as needed

try:
    asyncio.run(app())
except KeyboardInterrupt:
    pass
