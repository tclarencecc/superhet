import functools
from starlette.requests import Request
from starlette.routing import Route

def route(method: str, path: str):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*arg, **kwargs):
            async def endpoint(req: Request):
                params = tuple(req.path_params[k] for k in req.path_params.keys())
                ret = await fn(req, *params)
                return ret

            return Route(path, endpoint, methods=[method])
        return wrapper
    return decorate
