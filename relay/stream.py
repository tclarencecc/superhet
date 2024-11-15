import asyncio

from common.data import Answer
from relay.config import Config

class AnswerStream:
    def __init__(self):
        self._future: asyncio.Future = None
        self._end = False

    def update(self, ans: Answer):
        self._end = ans.end
        if not self._future.done() and not self._future.cancelled():
            self._future.set_result(ans.word)

    def cancel(self):
        self._end = True
        if not self._future.done() and not self._future.cancelled():
            self._future.set_result("")

    async def generator(self):
        while True:
            self._future = asyncio.get_event_loop().create_future()
            ret = await self._future
            yield ret
            if self._end:
                if Config.DEBUG:
                    print("answer stream ended")
                break
