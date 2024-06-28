from enum import Enum

class SplitType(Enum):
    CHAPTER = 0
    PARAGRAPH = 1
    SENTENCE = 2

def split(input: str, type: SplitType) -> list[str]:
    if type == SplitType.CHAPTER:
        # TODO add more chapter splitter
        # for k, v in kwargs.items():
        #     if k == ""
        return [input]

    paras = input.split("\n\n")
    if type == SplitType.PARAGRAPH:
        return paras
    
    ret = list[str]
    for para in paras:
        sens = para.split(".")
        for sen in sens:
            s = sen.strip() + "." #'.' was used as separator, re-add here
            ret.append(s)

    return ret
