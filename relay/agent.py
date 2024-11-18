from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocket
from starlette.routing import WebSocketRoute

from common.data import DataType, Notification, Answer
from common.serde import parse_type
from common.asynch import StreamToGenerator
from relay.config import Config

class _AgentRoute(WebSocketEndpoint):
    async def on_connect(self, websocket: WebSocket):    
        if Agents.connect(websocket):
            await websocket.accept()

            notif = Notification()
            notif.message = "Connected to relay"
            await websocket.send_json(notif.json())
        else:
            # closing here sends HTTP 403 to agent, which disables connection retry
            await websocket.close()

    async def on_receive(self, websocket: WebSocket, data: any):
        if parse_type(data, DataType) == DataType.ANSWER:
            ans = Answer(data)
            st = Agents.stream(websocket.headers[Config.HEADER.NAME], ans.id)
            st.get().update(ans.word, ans.end)

            if ans.end:
                st.delete()

    async def on_disconnect(self, websocket: WebSocket, close_code: int):
        Agents.disconnect(websocket)

agent_route = WebSocketRoute("/ws", _AgentRoute)


class Agents:
    _instance = None

    @staticmethod
    def _dict() -> dict:
        if Agents._instance is None:
            Agents._instance = {}
        return Agents._instance
    
    @staticmethod
    def has(name: str) -> bool:
        return Agents._dict().get(name, None) is not None
    
    @staticmethod
    def connect(websocket: WebSocket):
        if Config.API_KEY is not None:
            if Config.API_KEY != websocket.headers[Config.HEADER.KEY]:
                return False

        name = websocket.headers[Config.HEADER.NAME]
        # must have name and name must be unique
        if name is None or name == "" or Agents.has(name):
            return False

        Agents._dict()[name] = {
            "websocket": websocket,
            "streams": {}
        }
        Agents._debug()
        return True

    @staticmethod
    def disconnect(websocket: WebSocket):
        name = websocket.headers[Config.HEADER.NAME]
        # disconnect still gets called even if connect is not websocket.accept'ed!
        if not Agents.has(name):
            return

        # cancel all available (running) streams
        streams: dict = Agents._dict()[name]["streams"]
        for s in streams.keys():
            stg: StreamToGenerator = streams[s]
            stg.cancel()

        del Agents._dict()[name]
        Agents._debug()

    @staticmethod
    def websocket(name: str) -> WebSocket:
        return Agents._dict()[name]["websocket"]
    
    @staticmethod
    def _debug():
        if Config.DEBUG:
            print(f"Agents: {list(Agents._dict().keys())}")

    
    class _Stream:
        def __init__(self, name: str, id: str):
            self._name = name
            self._streams: dict = Agents._dict()[name]["streams"]
            self._id = id

        def new(self) -> StreamToGenerator:
            stg = StreamToGenerator(Config.DEBUG)
            self._streams[self._id] = stg
            self._debug()
            return stg
        
        def get(self) -> StreamToGenerator:
            return self._streams[self._id]
        
        def delete(self):
            del self._streams[self._id]
            self._debug()

        def _debug(self):
            if Config.DEBUG:
                print(f"'{self._name}' streams: {list(self._streams.keys())}")
        
    @staticmethod
    def stream(name: str, id: str) -> _Stream:
        return Agents._Stream(name, id)
