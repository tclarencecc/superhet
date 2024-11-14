import asyncio

from common.data import Answer

class AnswerStream:
    def __init__(self):
        self._future: asyncio.Future = None
        self._end = False

    def update(self, ans: Answer):
        self._end = ans.end
        if not self._future.done() and not self._future.cancelled():
            self._future.set_result(ans.word)

    async def generator(self):
        while True:
            self._future = asyncio.get_event_loop().create_future()
            ret = await self._future
            yield ret
            if self._end:
                break


class Streamers:
    _instance = None

    @staticmethod
    def _dict() -> dict:
        if Streamers._instance is None:
            Streamers._instance = {}
        return Streamers._instance

    @staticmethod
    def _has(id: str) -> bool:
        return Streamers._dict().get(id, None) is not None

    @staticmethod
    def add(id: str, anst: AnswerStream):
        if not Streamers._has(id):
            Streamers._dict()[id] = anst

    @staticmethod
    def get(id: str) -> AnswerStream | None:
        return Streamers._dict().get(id, None)
    
    @staticmethod
    def delete(id: str):
        if Streamers._has(id):
            del Streamers._dict()[id]
