import asyncio

from app.config import Config
from app.cli import cli
from app.storage import Sql
from app.llm import Completion, Embedding

async def main():
    def post_config_load():
        # setup non-toml based config values
        n_embd, n_ctx = Embedding.stats()
        Config.LLAMA.EMBEDDING.SIZE = n_embd
        Config.LLAMA.EMBEDDING.CONTEXT = n_ctx

    Config.load_from_toml(post_config_load)

    _ = Completion._llm() # log llama init now

    async with Sql():
        loop = asyncio.get_event_loop()
        loop.run_until_complete(await cli())
        # run more inf loops as needed

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
