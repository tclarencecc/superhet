from websockets.asyncio.client import connect

from agent.config import Config

async def server():
    async with connect(f"ws://{Config.SERVER.HOST}{Config.SERVER.ENDPOINT}") as ws:
        while True:
            msg = await ws.recv()
