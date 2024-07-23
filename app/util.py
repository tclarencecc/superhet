import time
import functools
import inspect
import subprocess
import shlex
import signal
import asyncio
from enum import Enum
from datetime import datetime, timezone
from typing import Callable

from app.config import Config

def _print_duration(name: str, t: float):
    t = time.time() - t
    if t >= 1:
        # 1.1 sec
        PrintColor.OK(f"{name}: {t:.1f} sec")
    elif t < 1 and t >= 0.1:
        # 0.11 sec
        PrintColor.OK(f"{name}: {t:.2f} sec")
    else: # < 0.1
        # 99 ms
        t = t * 1000
        PrintColor.OK(f"{name}: {t:.0f} ms")

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

def new_async_task(coro, callback: Callable=None):
    def _callback(task: asyncio.Task):
        if callback is not None:
            callback()

        try:
            # needed to avoid 'Task exception was never retrieved'
            # either returns None, exception or raise CancelledError, InvalidStateError
            ex = task.exception()
            if ex is not None:
                raise ex
            
        except Exception as e:
            print(e)

    task = asyncio.get_event_loop().create_task(coro)
    task.add_done_callback(_callback)

class EnumDict:
    def __init__(self):
        self._dict = {}

    def get(self, k: Enum) -> any:
        if self._dict.get(k.name) is None:
            self.set(k)
        
        return self._dict[k.name]

    def set(self, k: Enum, v: any=None):
        if v is not None:
            self._dict[k.name] = v
        else:
            if k.value is None:
                raise ValueError(f"EnumDict enum member '{k.name}' has no preset value. Set a user-defined value first.")

            self._dict[k.name] = k.value

class PrintColor:
    @staticmethod
    def OK(input: str):
        PrintColor._print("\033[92m", input)

    @staticmethod
    def WARN(input: str):
        PrintColor._print("\033[93m", input)

    @staticmethod
    def ERROR(input: str):
        PrintColor._print("\033[91m", input)

    @staticmethod
    def BLUE(input: str):
        PrintColor._print("\033[94m", input)

    @staticmethod
    def CYAN(input: str):
        PrintColor._print("\033[96m", input)

    @staticmethod
    def _print(color: str, input: str):
        print(f"{color}{input}\033[0m")
        
def timestamp() -> str:
    return datetime.now(
        datetime.now(timezone.utc).astimezone().tzinfo
    ).replace(microsecond=0).isoformat()
