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

CHAR_PAD = len(CHARS)
PHONEME_BOS = len(PHONEMES)
PHONEME_EOS = len(PHONEMES)
PHONEME_PAD = len(PHONEMES) + 1

def decode_chars(chars: List[int]) -> str:
    return "".join(map(lambda i: CHARS[i], chars))

def decode_phonemes(phons: List[int]) -> str:
    return " ".join(map(lambda i: PHONEMES[i], phons))

def to_kana(phons: List[int]) -> str:
    vowels = {
        "AA": 0,
        "AE": 0,
        "AH": 0,
        "AO": 5,
        
    }
