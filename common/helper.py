import time
from datetime import datetime, timezone
import asyncio
from typing import Coroutine, Callable

class PrintColor:
    @staticmethod
    def OK(input: str, stream=False):
        PrintColor._print("\033[92m", input, stream)

    @staticmethod
    def WARN(input: str, stream=False):
        PrintColor._print("\033[93m", input, stream)

    @staticmethod
    def ERROR(input: str, stream=False):
        PrintColor._print("\033[91m", input, stream)

    @staticmethod
    def BLUE(input: str, stream=False):
        PrintColor._print("\033[94m", input, stream)

    @staticmethod
    def CYAN(input: str, stream=False):
        PrintColor._print("\033[96m", input, stream)

    @staticmethod
    def _print(color: str, input: str, stream: bool):
        if stream:
            print(f"{color}{input}\033[0m", end="", flush=True)
        else:
            print(f"{color}{input}\033[0m")

def print_duration(name: str, t: float):
    t = time.time() - t
    if t >= 1:
        # 1.1 sec
        PrintColor.OK(f"{name}: {t:.1f} sec")
    elif t < 1 and t >= 0.1:
        # 0.11 sec
        PrintColor.OK(f"{name}: {t:.2f} sec")
    elif t < 0.1 and t >= 0.001:
        # 99 ms
        t = t * 1000
        PrintColor.OK(f"{name}: {t:.0f} ms")
    else: # < 0.001
        # 999 μs
        t = t * 1000000
        PrintColor.OK(f"{name}: {t:.0f} μs")

def timestamp() -> str:
    return datetime.now(
        datetime.now(timezone.utc).astimezone().tzinfo
    ).replace(microsecond=0).isoformat()

# https://gist.github.com/harrisont/38ecc65aaad3481c9221417d7c64fef8
def create_task(coro: Coroutine, loop: asyncio.AbstractEventLoop) -> Callable:
    async def error_handled_coro():
        try:
            return await coro()
        except Exception as e:
            print(e)

        # on initial SIGINT, KeyboardInterrupt is NOT raised
        # CancelledError is raised after task.cancel
        # KeyboardInterrupt is raised when run_until_complete is still running & SIGINT is called again
        except (asyncio.CancelledError, KeyboardInterrupt):
            # do nothing as task graceful shutdown is already handled
            pass

    task = loop.create_task(error_handled_coro())

    def cancel_handler():
        task.cancel() # on KeyboardInterrupt, task is not yet cancelled. cancel now
        loop.run_until_complete(task) # allow to complete then exit
    return cancel_handler
