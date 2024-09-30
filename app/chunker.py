from typing import Iterable

from app.stream import FileStream
from app.config import Config
from app.util import MutableString
from app.decorator import benchmark

class Chunker:
    def __init__(self, input: str, params: dict[str, any]={}):
        """
        params:
            size: int = word count per chunk
            overlap: float = decimal percentage of overlap in words between chunks
            alphabet: bool = is document encoded in purely alphabetical script? default true
            separator: str = document content separator. default \\n\\n
        """
        def assign(k: str, dv: any):
            if k in params:
                return params[k]
            else:
                return dv
                
        self._chunk_size = assign("size", Config.CHUNK.SIZE.MIN)
        self._chunk_overlap = assign("overlap", Config.CHUNK.OVERLAP.MIN)
        self._alphabet = assign("alphabet", True)
        separator = assign("separator", "\n\n")

        self._splitted = []

        if input.startswith("./"):
            self._iterable: Iterable[str] = FileStream(input, separator=separator)
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
                    _sliding_window(
                        next(self._iterable), 
                        self._chunk_size,
                        self._chunk_overlap,
                        self._alphabet
                    )
                )

        return self._splitted.pop(0)

def _split_to_sentence_weight(input: str, alphabet: bool) -> list[tuple[str, int]]:
    stop_marks = "!?."
    if alphabet == False:
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
            if alphabet:
                # >1 whitespaces will also count as 'words'. +1 for stop mark
                count = sentence.split_len(" ") + 1
            else:
                count = len(sentence) # whitespaces in between also count as 'word/s'

            ret.append((sentence.value(), count))
            sentence.clear()

    return ret

# https://qwen.readthedocs.io/en/latest/
# > Stable support of 32K context length for models of all sizes and
# > up to 128K tokens with Qwen2-7B-Instruct and Qwen2-72B-Instruct

def _sliding_window(input: str, chunk_size: int, overlap: float, alphabet: bool) -> list[str]:
    if overlap < Config.CHUNK.OVERLAP.MIN or overlap > Config.CHUNK.OVERLAP.MAX:
        raise ValueError(f"chunker._sliding_window overlap should be "
            f"between {Config.CHUNK.OVERLAP.MIN} and {Config.CHUNK.OVERLAP.MAX}.")
    
    min_cs = Config.CHUNK.SIZE.MIN
    if alphabet:
        # assuming a generous 2 token-per-word
        max_cs = int(Config.LLAMA.EMBEDDING.CONTEXT / 2)
    else:
        # multiple chars can be just 1 token; assume worst case 1 token-per-char with small allowance
        max_cs = int(Config.LLAMA.EMBEDDING.CONTEXT * 0.8)
        
    if chunk_size < min_cs or chunk_size > max_cs:
        raise ValueError(f"chunker._sliding_window chunk_size should be between {min_cs} and {max_cs}.")

    ret = []
    overlap_size = chunk_size * overlap

    snt_wgt = _split_to_sentence_weight(input, alphabet)
    idx = 0
    total = 0
    sentence = MutableString()

    while idx < len(snt_wgt):
        if alphabet and sentence.not_empty():
            sentence.add(" ")
        sentence.add(snt_wgt[idx][0])
        total += snt_wgt[idx][1]

        if total > chunk_size:
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
