import asyncio
from enum import Enum
from typing import Callable

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
