import asyncio
import certifi
import ssl
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidURI, InvalidHandshake, ConnectionClosed

from agent.config import Config
from agent.storage import Vector
from agent.llm import Embedding, Completion, Chat
from common.serde import parse_type
from common.data import DataType, Notification, Query, Answer, Request_File, Send_File
from common.helper import PrintColor
from common.iter import EndDefIter, EndDefFile

async def server():
    # TODO load chat history on per user basis
    chat = Chat()

    headers = [
        (Config.RELAY.HEADER.NAME, Config.RELAY.AGENT_NAME),
        (Config.RELAY.HEADER.KEY, Config.RELAY.API_KEY),
    ]
    protocol = "wss://" if Config.RELAY.ENABLE_TLS else "ws://"
    ssl_ctx = ssl.create_default_context(cafile=certifi.where()) if Config.RELAY.ENABLE_TLS else None

    try:
        # connect() already does exponential backoff on connection retries
        # https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html#
        async for ws in connect(f"{protocol}{Config.RELAY.HOST}{Config.RELAY.ENDPOINT}",
            additional_headers=headers,
            ssl=ssl_ctx):
            
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

                    elif dt == DataType.REQUEST_FILE:
                        rf = Request_File(json_str=msg)
                        if rf.file_type == "html":
                            path = Config.RELAY.HTML_APP_PATH
                        else:
                            # no other type for now
                            assert rf.file_type == "html"
                        
                        with EndDefFile(path, Config.RELAY.HTML_SERVE_SIZE) as res:
                            for r, end in res:
                                sf = Send_File()
                                sf.id = rf.id
                                sf.part = r # base64.b64encode(r)
                                sf.binary = False
                                sf.end = end

                                await ws.send(sf.json_string())

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
