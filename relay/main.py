from starlette.applications import Starlette

from relay.config import Config
from relay.agent import agent_route
from relay.api import html_route, query_route

Config.load()

app = Starlette(routes=(
    html_route,
    query_route,
    agent_route
))
