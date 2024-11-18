from typing import Iterator
from io import TextIOWrapper, BufferedReader

class EndDefIter:
    def __init__(self, itr: Iterator):
        self._iterable = itr
        self._curr = None
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
        
        ret = self._curr
        try:
            self._curr = next(self._iterable)
        except StopIteration:
            self._end = True

        return (ret, self._end)


class EndDefFile:
    class _Iter:
        def __init__(self, file: TextIOWrapper | BufferedReader, size: int):
            self._file: TextIOWrapper | BufferedReader = file
            self._size = size
            self._end = False

        def __iter__(self):
            return self

        def __next__(self) -> tuple[any, bool]:
            if self._end:
                raise StopIteration
            
            # TODO edge case: if last chunk of file read is exactly equal to self.size,
            # end will still be false but on next iteration read will be totally empty
            curr = self._file.read(self._size)
            if len(curr) < self._size:
                self._end = True

            return (curr, self._end)

    def __init__(self, path: str, size: int, binary=False):
        self._file: TextIOWrapper | BufferedReader = None
        self._path = path
        self._size = size
        self._binary = binary

    def __enter__(self):
        self._file = open(self._path, "rb" if self._binary else "r")
        return EndDefFile._Iter(self._file, self._size)
        
    def __exit__(self, type, value, traceback):
        self._file.close()
