from starlette.applications import Starlette

from relay.agent import agent_route
from relay.api import html, query
from relay.config import Config

Config.load()

app = Starlette(routes=(
    html(),
    query(),
    agent_route
))
