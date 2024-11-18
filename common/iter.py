from typing import Iterator

class EndDefIter:
    def __init__(self, itr: Iterator):
        self._iterable = itr
        self._curr = None
        self._next = None
        self._end = False

        try:
            self._curr = next(self._iterable)
        except StopIteration:
            self._end = True

    def __iter__(self):
        return self

    def __next__(self) -> tuple[any, bool]:
        if self._end:
            raise StopIteration
        
        try:
            self._next = next(self._iterable)
        except StopIteration:
            self._end = True

        ret = self._curr
        self._curr = self._next

        return (ret, self._end)
