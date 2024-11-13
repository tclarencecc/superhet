
class MutableString:
    def __init__(self):
        self._buffer = [""] * 100
        self._idx = -1

    def add(self, input: str):
        if (len(self) + len(input)) > len(self._buffer):
            # extend by input len though this might be more than whats needed
            self._buffer.extend([""] * len(input))

        for char in input:
            self._idx += 1
            self._buffer[self._idx] = char

    def clear(self):
        self._idx = -1

    def value(self):
        # array slice end idx IS excluded
        return "".join(self._buffer[:self._idx + 1])
    
    def not_empty(self) -> bool:
        return self._idx > -1
    
    def strip(self):
        txt = self.value().strip()
        if len(txt) != len(self):
            self.clear()
            self.add(txt)

    def split_len(self, char: str) -> int:
        if len(char) != 1:
            raise ValueError(f"MutableString.split_len only accepts character splitter. Splitter used '{char}'.")
        
        ret = 1 # count current unsplitted text as already 1 partition
        for i in range(len(self)):
            if self._buffer[i] == char:
                ret += 1
        return ret

    def __len__(self) -> int:
        return self._idx + 1
