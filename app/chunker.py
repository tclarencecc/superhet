from typing import Iterable
from io import StringIO

from app.config import Config, DocumentScript
from common.string import MutableString

class Chunker:
    def __init__(self, input: str):
        self._splitted = []

        if input.startswith("./"):
            self._iterable: Iterable[str] = FileStream(input)
        elif input.startswith("<!DOCTYPE html>"):
            # TODO should be handling http stream here
            raise NotImplementedError
        else:
            # document as string is not allowed
            raise ValueError("Chunker only accepts file path or http stream inputs.")

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._splitted) == 0:
            # repeat call on next(_iterable) until _splitted is filled up
            # once _iterable is emptied, StopIteration will be raised and bubbled up
            while len(self._splitted) == 0:
                self._splitted.extend(
                    _sliding_window(next(self._iterable))
                )

        return self._splitted.pop(0)

def _split_to_sentence_weight(input: str) -> list[tuple[str, int]]:
    if Config.CHUNK.SCRIPT == DocumentScript.LATIN:
        stop_marks = "!?."
    elif Config.CHUNK.SCRIPT == DocumentScript.HANZI:
        stop_marks = "！？｡。"

    ret = []
    idx = 0
    max = len(input)
    sentence = MutableString()

    for char in input:
        sentence.add(char)
        idx += 1

        if char in stop_marks:
            # if ".", look ahead in case is part of a 'x.x' word
            if char == "." and idx < max: # idx already points to next elem!
                if input[idx] != " " and input[idx] != "\n":
                    continue # skip outputting; is part of 'x.x' word

            sentence.strip()
            if Config.CHUNK.SCRIPT == DocumentScript.LATIN:
                # >1 whitespaces will also count as 'words'. +1 for stop mark
                count = sentence.split_len(" ") + 1
            elif Config.CHUNK.SCRIPT == DocumentScript.HANZI:
                count = len(sentence) # whitespaces in between also count as 'word/s'

            ret.append((sentence.value(), count))
            sentence.clear()

    return ret

def _sliding_window(input: str) -> list[str]:
    ret = []
    overlap_size = Config.CHUNK.SIZE * Config.CHUNK.OVERLAP

    snt_wgt = _split_to_sentence_weight(input)
    idx = 0
    total = 0
    sentence = MutableString()

    while idx < len(snt_wgt):
        if Config.CHUNK.SCRIPT == DocumentScript.LATIN and sentence.not_empty():
            sentence.add(" ")
        sentence.add(snt_wgt[idx][0])
        total += snt_wgt[idx][1]

        if total > Config.CHUNK.SIZE:
            # sentence collected up to this point is enough, output it
            ret.append(sentence.value())

            # apply sliding window; slide back overlap% reusing previous sentences
            deduct = 0
            while True:
                deduct += snt_wgt[idx][1] # start reusing current idx
                if deduct >= overlap_size:
                    break
                idx = idx - 1

            total = 0
            sentence.clear()
        else:
            idx += 1

    # sentence outputting happens when chunk_size is reached (handled above) OR
    # looping through all sentences has completed and chunk_size is not yet reached
    if sentence.not_empty():
        ret.append(sentence.value())

    return ret


class FileStream:
    def __init__(self, file: str):
        # do NOT make file async (aiofiles) as this makes the chain an async-iter all the way:
        # FileStream -> Chunker -> Embedding
        # Embedding is not an async function so it becomes blocking I/O in the end
        self._reader = open(file) # let errors go through
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
        separator = Config.CHUNK.SEPARATOR
        count = len(separator)
        buffer = [""] * count
        rslen = len(self._read_str)

        def flush():
            nonlocal idx
            if idx == 0:
                # most of the time char read IS NOT part of separator so just write [0]
                writer.write(buffer[0])
            else:
                if buffer[idx] == separator[0]:
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

                if buffer[idx] == separator[idx]:
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
