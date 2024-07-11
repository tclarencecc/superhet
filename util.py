import time
import functools
import inspect
import subprocess
import shlex
import signal
from config import Config

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
        if Config.BENCHMARK == False:
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

def extprocess(args: list[tuple[str, str]]):
    """
    args: list of tuple (cwd: str, cmd: str, env: dict)
    env is optional
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper():
            procs = []
            try:
                for arg in args:
                    env = None
                    if len(arg) == 3:
                        # has env
                        env = arg[2]

                    procs.append(subprocess.Popen(
                        shlex.split(arg[1]),
                        cwd=arg[0],
                        env=env
                    ))
                fn()
            except KeyboardInterrupt:
                # SIGINT propagates to child processes, do nothing
                pass
            except Exception as e:
                # exceptions raised from long-running loop will be catched here
                for proc in procs:
                    proc.send_signal(signal.SIGINT)
                raise e
            finally:
                pass

        return wrapper
    return decorate
