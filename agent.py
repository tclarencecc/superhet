import asyncio

from agent.config import Config
from agent.cli import cli
from agent.server import server
from agent.storage import Sql
from agent.llm import Completion, Embedding
from common.helper import create_task

def post_config_load():
    # setup non-toml based config values
    n_embd, n_ctx = Embedding.stats()
    Config.LLAMA.EMBEDDING.SIZE = n_embd
    Config.LLAMA.EMBEDDING.CONTEXT = n_ctx

Config.load_from_toml(post_config_load)

_ = Completion._llm() # log llama init now

with Sql():
    loop = asyncio.new_event_loop()
    t1 = create_task(cli, loop)
    t2 = create_task(server, loop)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        t1()
        t2()
    finally:
        loop.close()
