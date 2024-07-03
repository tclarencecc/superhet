from typing import Iterable
from stream import FileStream

class Chunker:
    def __init__(self, input: str, params: dict[str, any]={}):
        def assign(k: str, dv: any):
            if k in params:
                return params[k]
            else:
                return dv

        self._chunk_size = assign("chunk_size", 100)
        self._chunk_overlap = assign("chunk_overlap", 0.25)
        self._alphabet = assign("alphabet", True)
        separator = assign("separator", "\n\n")

        self._splitted = []

        if input.startswith("./"):
            self._iterable: Iterable[str] = FileStream(input, separator=separator)
        elif input.startswith("<!DOCTYPE html>"):
            # TODO
            pass
        else:
            # TODO
            pass

    def __iter__(self):
        return self

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
    sentence = "" # no need for 'stringwriter'; each assembled sentence is just a subsection of a paragraph

    for char in input:
        sentence += char
        idx += 1

        if char in stop_marks:
            # if ".", look ahead in case is part of a 'x.x' word
            if char == "." and idx < max: # idx already points to next elem!
                if input[idx] != " " and input[idx] != "\n":
                    continue # skip outputting; is part of 'x.x' word

            sentence = sentence.strip()
            if alphabet:
                # >1 whitespaces will also count as 'words'. +1 for stop mark
                count = len(sentence.split(" ")) + 1
            else:
                count = len(sentence) # whitespaces in between also count as 'word/s'

            ret.append((sentence, count))
            sentence = ""

    return ret

# https://qwen.readthedocs.io/en/latest/
# > Stable support of 32K context length for models of all sizes and
# > up to 128K tokens with Qwen2-7B-Instruct and Qwen2-72B-Instruct
# setting chunk_size:
# if alphabet=true (latin), assuming a generous 2 token-per-word, chunk is 32k / 2 = 16k
# if false (hanzi), multiple chars can be just 1 token; assume worst case 1 token-per-char, chunk is 32k 

def _sliding_window(input: str, chunk_size: int, overlap: float, alphabet: bool) -> list[str]:
    if overlap < 0.1 or overlap > 0.5:
        raise Exception("chunker.split overlap must be 0.1 to 0.5.")
    if chunk_size < 100:
        raise Exception("chunker.split chunk_size should be at least 100.")

    ret = []
    overlap_size = chunk_size * overlap

    snt_wgt = _split_to_sentence_weight(input, alphabet)
    idx = 0
    total = 0
    sentence = "" # no need for 'stringwriter'; each assembled sentence is just a subsection of a paragraph

    while idx < len(snt_wgt):
        if alphabet and sentence != "":
            sentence += " "
        sentence += snt_wgt[idx][0]
        total += snt_wgt[idx][1]

        if total > chunk_size:
            # sentence collected up to this point is enough, output it
            ret.append(sentence)

            # apply sliding window; slide back overlap% reusing previous sentences
            deduct = 0
            while True:
                deduct += snt_wgt[idx][1] # start reusing current idx
                if deduct >= overlap_size:
                    break
                idx = idx - 1

            total = 0
            sentence = ""
        else:
            idx += 1

    # sentence outputting happens when chunk_size is reached (handled above) OR
    # looping through all sentences has completed and chunk_size is not yet reached
    if sentence != "":
        ret.append(sentence)

    return ret
