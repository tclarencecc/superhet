import asyncio
import base64
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidURI, InvalidHandshake, ConnectionClosed

from agent.config import Config
from agent.storage import Vector
from agent.llm import Embedding, Completion, Chat
from common.serde import parse_type
from common.data import DataType, Notification, Query, Answer
from common.helper import PrintColor
from common.iter import EndDefIter

async def server():
    # TODO load chat history on per user basis
    chat = Chat()

    headers = [
        (Config.RELAY.HEADER.NAME, Config.RELAY.AGENT_NAME),
        (Config.RELAY.HEADER.KEY, Config.RELAY.API_KEY),
    ]

    try:
        # connect() already does exponential backoff on connection retries
        # https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html#
        async for ws in connect(f"ws://{Config.RELAY.HOST}{Config.RELAY.ENDPOINT}",
            additional_headers=headers):
            
            # set name for helper.create_task.cancel_handler!
            ws.keepalive_task.set_name("keepalive")

            try:
                while True:
                    msg = await ws.recv()
                    dt = parse_type(msg, DataType)

                    if dt == DataType.NOTIFICATION:
                        print(Notification(json_str=msg).message)

                    elif dt == DataType.QUERY:
                        query = Query(json_str=msg)
                        vec = Embedding.from_string(query.text)
                        ctx = Vector.read(vec)
                        res = Completion.run(query.text, ctx, chat)

                        print(f"{query.text}\n")
                    
                        for r, end in EndDefIter(res):
                            ans = Answer()
                            ans.id = query.id
                            ans.word = r
                            ans.end = end

                            await ws.send(ans.json_string())
                            PrintColor.BLUE(r, stream=True)
                            if end:
                                print("\n")

            # reconnect-enabled error
            except ConnectionClosed:
                print(f"Disconnected from relay ({Config.RELAY.HOST})")

            # without contextib ('with' wrapper), manually handle closing of websocket
            # re-raising CancelledError exits connect-retry loop
            except asyncio.CancelledError as e:
                await ws.close()
                raise e

    # no reconnect for these errors
    except (InvalidURI, OSError):
        print(f"Unable to find relay at {Config.RELAY.HOST}")
    except (InvalidHandshake, TimeoutError):
        print(f"Connection refused by relay ({Config.RELAY.HOST})")
