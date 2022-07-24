import string
from typing import List

CHARS = list(string.ascii_uppercase)
char2id = {k: i for i, k in enumerate(CHARS)}

"""ARPABET Symbols"""
PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B",  "CH", "D",  "DH", "EH", "ER",
    "EY", "F",  "G",  "HH", "IH", "IY",
    "JH", "K",  "L",  "M",  "N",  "NG",
    "OW", "OY", "P",  "R",  "S",  "SH",
    "T",  "TH", "UH", "UW", "V",  "W",
    "Y",  "Z",  "ZH"
]
phon2id = {k: i for i, k in enumerate(PHONEMES)}

arpa_index_to_ojt = [
    "a", "a", "a", "o o", "a u", "a i",
    "b",  "ch", "d",  "z", "e", "a a",
    "e i", "f",  "g",  "h", "i", "i i",
    "j", "k",  "r",  "m",  "n",  "N g",
    "o u", "o i", "p",  "r",  "s",  "sh",
    "t",  "s", "u", "u u", "v",  "w",
    "y",  "z",  "z"
]
arpa_consonants = [
    "B",  "CH", "D",  "DH",
    "F",  "G",  "HH",
    "JH", "K",  "L",  "M",  "N",  "NG",
    "P",  "R",  "S",  "SH",
    "T",  "TH", "V",  "W",
    "Y",  "Z",  "ZH"
]

CHAR_PAD = len(CHARS)
PHONEME_BOS = len(PHONEMES)
PHONEME_EOS = len(PHONEMES)
PHONEME_PAD = len(PHONEMES) + 1

def decode_chars(chars: List[int]) -> str:
    return "".join(map(lambda i: CHARS[i], chars))

def decode_phonemes(phons: List[int]) -> str:
    return " ".join(map(lambda i: PHONEMES[i], phons))

def to_kana(phons: List[int]) -> str:
    from mora_list import openjtalk_mora2text
    out = ""
    state = ""
    for i in phons:
        p = PHONEMES[i]
        if state == "":
            if p in arpa_consonants:
                out += arpa_index_to_ojt[i]
                state = p
            else:
                out += arpa_index_to_ojt[i]
                out += " "
                state = ""
        else:
            if p in arpa_consonants:
                # add unvoice for previous consonant
                if state == "N":
                    out = out[:-1] + "N "
                elif state == "T":
                    out += "o "
                else:
                    out += "u "
                out += arpa_index_to_ojt[i]
                state = p
            else:
                out += arpa_index_to_ojt[i]
                out += " "
                state = ""

    if state != "":
        # add unvoice for previous consonant
        if state == "N":
            out = out[:-1] + "N "
        elif state == "T":
            out += "o "
        else:
            out += "u "

    out = out.strip().split(" ")
    return "".join([openjtalk_mora2text[text] for text in out])
