import uuid
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from common.data import Query
from relay.agent import Agents
from relay.decorator import route

@route("GET", "/api/{agent:str}")
async def _query(req: Request, agent: str):
    query = Query()
    query.id = uuid.uuid4().hex
    query.text = "what is crit rate"

    # agent cannot be none or empty as routing would not match
    if Agents.has(agent):
        ws = Agents.websocket(agent)
        anst = Agents.stream(agent, query.id).new()
        
        await ws.send_json(query.json())
        return StreamingResponse(anst.generator())
    else:
        return Response("Agent unavailable")

query_route = _query()
