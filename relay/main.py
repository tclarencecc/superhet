from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from relay.config import Config
from relay.agent import agent_route
from relay.api import query_route

Config.load()

app = Starlette(routes=(
    #Mount("/", app=StaticFiles(directory="/relay")),
    query_route,
    agent_route
))
