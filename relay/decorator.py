import functools
import traceback
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from typing import Callable
from common.helper import PrintColor

def route(method: str, path: str, middlewares: list[Callable[[Request, any], any]]=None):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*arg, **kwargs):
            async def endpoint(req: Request):
                params = tuple(req.path_params[k] for k in req.path_params.keys())

                if middlewares is not None:
                    for mw in middlewares:
                        ret = mw(req, *params)
                        if ret is not None:
                            return ret
                
                try:
                    ret = await fn(req, *params)
                    return ret
                except Exception:
                    PrintColor.ERROR(traceback.format_exc())
                    return Response(status_code=500)

            return Route(path, endpoint, methods=[method])
        return wrapper
    return decorate
