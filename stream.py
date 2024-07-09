from io import StringIO

class FileStream:
    def __init__(self, file: str, separator="\n\n"):
        if separator == "":
            raise ValueError("FileStream separator is required.")
        
        # do NOT make file async (aiofiles) as this makes FileStream an async iterable
        # in turn, Chunker also becomes async iterable. however, qdrant async client
        # does NOT support iterating through async iterables!
        self._reader = open(file) # let errors go through
        self._separator = separator
        self._eof = False

        self._read_str = ""
        self._read_idx = -1
        
    def __iter__(self):
        return self

    def __next__(self):
        if self._eof:
            raise StopIteration

        writer = StringIO()
        idx = 0
        count = len(self._separator)
        buffer = [""] * count
        rslen = len(self._read_str)

        def flush():
            nonlocal idx
            if idx == 0:
                # most of the time char read IS NOT part of separator so just write [0]
                writer.write(buffer[0])
            else:
                if buffer[idx] == self._separator[0]:
                    # handle edge case where latest char breaks similarity
                    # but is actually the beginning of a new separator
                    writer.write("".join(buffer[:idx]))
                    buffer[0] = buffer[idx]
                    idx = 1
                else:
                    writer.write("".join(buffer[:idx + 1]))
                    idx = 0

        while True:
            # init condition (idx -1) or finished parsing '4096 char' chunk
            if self._read_idx == -1 or self._read_idx == rslen:
                self._read_str = self._reader.read(2048) # 11 bit length
                self._read_idx = 0
                rslen = len(self._read_str) # re-establish read len

            # _reader.read can return only "" EOF (last read already exhausted all content)
            if self._read_str != "":
                buffer[idx] = self._read_str[self._read_idx]
                self._read_idx += 1

                if buffer[idx] == self._separator[idx]:
                    if (idx + 1) == count: # similar all the way to last char
                        break
                    else:
                        idx += 1
                else:
                    flush() # confirmed not similar
            else:
                flush()
                self._eof = True
                self._reader.close()
                break
            
        ret = writer.getvalue().strip()

        # handle edge case where separator is before EOF (w/ whitespaces in between)
        if ret == "" and self._eof:
            raise StopIteration
        
        return ret
