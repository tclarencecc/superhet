# module name is asynch bec pylance cant resolve async as module name!

from asyncio import Future, AbstractEventLoop, all_tasks, get_event_loop, CancelledError
from typing import Coroutine, Callable

# https://gist.github.com/harrisont/38ecc65aaad3481c9221417d7c64fef8
def create_task(coro: Coroutine, loop: AbstractEventLoop) -> Callable:
    async def error_handled_coro():
        try:
            return await coro()
        except Exception as e:
            print(e)

        # on initial SIGINT, KeyboardInterrupt is NOT raised
        # CancelledError is raised after task.cancel
        # KeyboardInterrupt is raised when run_until_complete is still running & SIGINT is called again
        except (CancelledError, KeyboardInterrupt):
            # do nothing as task graceful shutdown is already handled
            pass

    task = loop.create_task(error_handled_coro())
    task.set_name(coro.__name__) # set name so cancel_handler can distinguish between "my" tasks and system tasks!

    def cancel_handler():
        task.cancel() # on KeyboardInterrupt, task is not yet cancelled. cancel now
        loop.run_until_complete(task) # allow task to complete then exit

        # catch any system tasks left behind and let them run to completion
        for t in all_tasks(loop):
            if t.get_name().startswith("Task-"):
                loop.run_until_complete(t)
        
    return cancel_handler

class StreamToGenerator:
    def __init__(self, debug=False):
        self._future: Future = None
        self._end = False
        self._debug = debug

    def update(self, res: any, end: bool):
        self._end = end
        if not self._future.done() and not self._future.cancelled():
            self._future.set_result(res)

    def cancel(self):
        # cancelled; output stream is incomplete, just append an empty string regardless of type
        self.update("", True)

    async def generator(self):
        while True:
            self._future = get_event_loop().create_future()
            ret = await self._future
            yield ret
            if self._end:
                if self._debug:
                    print("StreamToGenerator ended")
                break
