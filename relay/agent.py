from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocket

from common.data import DataType, Notification, Answer
from common.serde import parse_type
from relay.stream import AnswerStream

class AgentEndpoint(WebSocketEndpoint):
    async def on_connect(self, websocket: WebSocket):    
        if Agents.connect(websocket):
            await websocket.accept()

            notif = Notification()
            notif.message = "Connected to relay"
            await websocket.send_json(notif.json())
        else:
            await websocket.close()

    async def on_receive(self, websocket: WebSocket, data: any):
        if parse_type(data, DataType) == DataType.ANSWER:
            ans = Answer(data)
            st = Agents.stream(websocket.headers["Agent-Name"], ans.id)
            st.get().update(ans)

            if ans.end:
                st.delete()

    async def on_disconnect(self, websocket: WebSocket, close_code: int):
        Agents.disconnect(websocket)


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
        # TODO check Agent-ApiKey

        name = websocket.headers["Agent-Name"]
        if name is None or name == "" or Agents.has(name):
            return False
        else:
            Agents._dict()[name] = {
                "websocket": websocket,
                "streams": {}
            }
            Agents._debug()
            return True

    @staticmethod
    def disconnect(websocket: WebSocket):
        name = websocket.headers["Agent-Name"]

        # cancel all available (running) streams
        streams: dict = Agents._dict()[name]["streams"]
        for s in streams.keys():
            anst: AnswerStream = streams[s]
            anst.cancel()

        del Agents._dict()[name]
        Agents._debug()

    @staticmethod
    def websocket(name: str) -> WebSocket:
        return Agents._dict()[name]["websocket"]
    
    @staticmethod
    def _debug():
        print(f"Agents: {list(Agents._dict().keys())}")

    
    class _Stream:
        def __init__(self, name: str, id: str):
            self._name = name
            self._streams: dict = Agents._dict()[name]["streams"]
            self._id = id

        def new(self) -> AnswerStream:
            anst = AnswerStream()
            self._streams[self._id] = anst
            self._debug()
            return anst
        
        def get(self) -> AnswerStream:
            return self._streams[self._id]
        
        def delete(self):
            del self._streams[self._id]
            self._debug()

        def _debug(self):
            print(f"'{self._name}' streams: {list(self._streams.keys())}")
        
    @staticmethod
    def stream(name: str, id: str) -> _Stream:
        return Agents._Stream(name, id)
