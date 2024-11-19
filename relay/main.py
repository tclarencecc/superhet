from starlette.applications import Starlette

from relay.config import Config
from relay.agent import agent_route
from relay.api import html, query

Config.load()

app = Starlette(routes=(
    html(),
    query(),
    agent_route
))
