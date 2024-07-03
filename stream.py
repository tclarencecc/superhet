from io import StringIO

class FileStream:
    def __init__(self, file: str, separator="\n\n"):
        if separator == "":
            raise Exception("Chunker separator is required.")
        
        self._reader = open(file) # let errors go through
        self._separator = separator
        self._eof = False
        
    def __iter__(self):
        return self

    def __next__(self):
        if self._eof:
            raise StopIteration

        writer = StringIO()
        idx = 0
        count = len(self._separator)
        buffer = [""] * count

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
            char = self._reader.read(1)
            buffer[idx] = char
            
            if char == "": # EOF
                flush()
                self._eof = True
                self._reader.close()
                break
            else:
                if buffer[idx] == self._separator[idx]:
                    if (idx + 1) == count: # similar all the way to last char
                        break
                    else:
                        idx += 1
                else:
                    flush() # confirmed not similar

        ret = writer.getvalue().strip()

        # handle edge case where separator is before EOF (w/ whitespaces in between)
        if ret == "" and self._eof:
            raise StopIteration
        
        return ret
