from websockets.asyncio.client import connect
from websockets.exceptions import InvalidURI, InvalidHandshake, ConnectionClosed

from agent.config import Config
from agent.storage import Vector
from agent.llm import Embedding, Completion, Chat
from common.serde import parse_type
from common.data import DataType, Notification, Query, Answer

async def server():
    # TODO load chat history on per user basis
    chat = Chat()

    headers = [
        (Config.RELAY.HEADER.NAME, Config.RELAY.AGENT_NAME),
    ]

    # connect() already does exponential backoff on connection retries
    # https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html#
    async for ws in connect(f"ws://{Config.RELAY.HOST}{Config.RELAY.ENDPOINT}",
        additional_headers=headers):
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
                
                    for r in res:
                        ans = Answer()
                        ans.id = query.id
                        ans.word = r
                        await ws.send(ans.json_string())

                    ans = Answer()
                    ans.id = query.id
                    ans.end = True
                    await ws.send(ans.json_string())
                    
        except (InvalidURI, OSError):
            print(f"Unable to connect to relay ({Config.RELAY.HOST})")
        except (InvalidHandshake, TimeoutError):
            print(f"Error establishing connection with relay ({Config.RELAY.HOST})")
        except ConnectionClosed:
            print(f"Disconnected from relay ({Config.RELAY.HOST})")
