import functools
import inspect
import time
import builtins

from common.helper import print_duration

# decorate() is called immed on import of module with @benchmark
# if run is derived from config, make sure it's already set before importing modules with @benchmark!
def benchmark(name: str, run: bool):
    def decorate(fn):
        if not run:
            return fn
        
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*arg, **kwargs):
                t = time.time()
                ret = await fn(*arg, **kwargs)
                print_duration(name, t)
                return ret
        else:
            @functools.wraps(fn)
            def wrapper(*arg, **kwargs):
                t = time.time()
                ret = fn(*arg, **kwargs)
                print_duration(name, t)
                return ret

        return wrapper
    return decorate

def suppress_print(prefixes: tuple[str, ...]):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*arg, **kwargs):
            old_print = builtins.print
            def new_print(values: str, sep=" ", end="\n", file=None, flush=False):
                 if not values.startswith(prefixes):
                    old_print(values, sep=sep, end=end, file=file, flush=flush)

            builtins.print = new_print
            ret = fn(*arg, **kwargs)
            builtins.print = old_print

            return ret

        return wrapper
    return decorate
