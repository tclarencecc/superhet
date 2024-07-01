import time
import functools
import config
from config import ConfigKey

# decor-fn is called immed on import of module with @decorator
# if decor-fn uses config, make sure config is set BEFORE import of said module!
def benchmark(name: str):
    def decorate(fn):
        if config.get(ConfigKey.BENCHMARK) == False:
            return fn
        
        @functools.wraps(fn)
        def wrapper(*arg, **kwargs):
            t = time.time()
            ret = fn(*arg, **kwargs)
            t = time.time() - t

            tf = "sec"
            if t < 0.1:
                t = t * 1000
                tf = "ms"

            print("\033[92m{name}: {t:.3f} {tf}\033[0m".format(name=name, t=t, tf=tf))
            return ret
        return wrapper
    return decorate
