import time
import functools
import inspect
import config
from config import ConfigKey

def _print_duration(name: str, t: float):
    t = time.time() - t
    tf = "sec"
    if t < 0.1:
        t = t * 1000
        tf = "ms"

    print("\033[92m{name}: {t:.3f} {tf}\033[0m".format(name=name, t=t, tf=tf))

# decor-fn is called immed on import of module with @decorator
# if decor-fn uses config, make sure config is set BEFORE import of said module!
def benchmark(name: str):
    def decorate(fn):
        if config.get(ConfigKey.BENCHMARK) == False:
            return fn
        
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*arg, **kwargs):
                t = time.time()
                ret = await fn(*arg, **kwargs)
                _print_duration(name, t)
                return ret
        else:
            @functools.wraps(fn)
            def wrapper(*arg, **kwargs):
                t = time.time()
                ret = fn(*arg, **kwargs)
                _print_duration(name, t)
                return ret

        return wrapper
    return decorate

class HttpError(Exception): ...
