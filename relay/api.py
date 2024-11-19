import uuid
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from common.data import Query, Request_File
from relay.agent import Agents
from relay.decorator import route

def _has_agent(req: Request, agent: str) -> Response | None:
    if not Agents.has(agent):
        return Response("Agent unavailable", status_code=404)


@route("POST", "/a/{agent:str}/query", middlewares=[_has_agent])
async def query(req: Request, agent: str):
    form = await req.json()
    query = Query()
    query.id = uuid.uuid4().hex
    query.text = form["query"]

    ws = Agents.websocket(agent)
    stg = Agents.stream(agent, query.id).new()
    
    await ws.send_json(query.json())
    return StreamingResponse(stg.generator())

@route("GET", "/a/{agent:str}", middlewares=[_has_agent])
async def html(req: Request, agent: str):
    rf = Request_File()
    rf.file_type = "html"
    rf.id = uuid.uuid4().hex

    ws = Agents.websocket(agent)
    stg = Agents.stream(agent, rf.id).new()
    
    await ws.send_json(rf.json())
    return StreamingResponse(stg.generator(), media_type="text/html")
