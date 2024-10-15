from typing import Iterable

from app.stream import FileStream
from app.config import Config, DocumentScript
from app.util import MutableString
from app.decorator import benchmark

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

    # @benchmark("chunker next")
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
