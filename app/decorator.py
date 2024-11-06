import functools
import inspect
import subprocess
import shlex
import signal
import time
import builtins

from app.config import Config
from app.util import print_duration

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

def extprocess(args: list[tuple[str, str]]):
    """
    args: list of tuple (cwd: str, cmd: str, env: dict)\n
    env is optional
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper():
            procs = []
            sigint = False

            try:
                for arg in args:
                    env = None
                    if len(arg) == 3:
                        # has env
                        env = arg[2]

                    stdout = None
                    stderr = None
                    if Config.PROCESS_STDOUT == False:
                        stdout = subprocess.DEVNULL
                        stderr = subprocess.DEVNULL

                    procs.append(subprocess.Popen(
                        shlex.split(arg[1]),
                        cwd=arg[0],
                        env=env,
                        stdout=stdout,
                        stderr=stderr
                    ))
                fn()
            except KeyboardInterrupt:
                # SIGINT propagates to child processes
                sigint = True
                pass
            finally:
                if sigint == False:
                    for proc in procs:
                        proc.send_signal(signal.SIGINT)
                
                # regardless of how parent proc ended, check child proc status
                # wait is a blocking call but parent proc has already ended by now anyway
                for proc in procs:
                    if proc.wait() != 0:
                        print(f"pid {proc.pid} did not terminate.")
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
