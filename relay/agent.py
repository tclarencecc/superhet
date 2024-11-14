from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocket

from common.data import DataType, Notification, Answer
from common.serde import parse_type
from relay.stream import Streamers

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
            anst = Streamers.get(ans.id)
            if anst is None:
                raise RuntimeError("on_receive accessing non-existent streamer")
            else:
                anst.update(ans)

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
    def _has(name: str) -> bool:
        return Agents._dict().get(name, None) is not None
    
    @staticmethod
    def get(name: str) -> WebSocket | None:
        return Agents._dict().get(name, None)

    @staticmethod
    def connect(websocket: WebSocket):
        # TODO check Agent-ApiKey

        name = websocket.headers["Agent-Name"]
        if name is None or name == "" or Agents._has(name):
            return False
        else:
            Agents._dict()[name] = websocket
            return True

    @staticmethod
    def disconnect(websocket: WebSocket):
        name = websocket.headers["Agent-Name"]
        if Agents._has(name):
            del Agents._dict()[name]
