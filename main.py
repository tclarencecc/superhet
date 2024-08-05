import asyncio
from httpx import AsyncClient

from app.util import extprocess
from app.config import Config
from app.cli import cli
import app.db as db

async def app():
    async with AsyncClient() as client:
        db.http_client(client)
        await cli()

@extprocess([
    (Config.QDRANT.PATH, Config.QDRANT.SHELL, Config.QDRANT.ENV)
    ])
def main():
    asyncio.run(app())

main()
