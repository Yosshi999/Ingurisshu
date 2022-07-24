import argparse
from pathlib import Path
import sys
import pickle
import re

from .common import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dic", type=str, default="./cmudict-0.7b.txt", help="Path to the CMU Dict")
    args = parser.parse_args()
    DIC = Path(args.dic)

    print("reading %s..." % str(DIC))
    data = DIC.read_text().split("\n")
    read = 0
    saved_words = []
    for line in data:
        read += 1
        line = line.strip()
        if read % 100 == 0 or read == len(data):
            sys.stdout.write("\rread %d/%d, saved %d words" % (read, len(data), len(saved_words)))
        if line.startswith(";;;"):
            # comment line
            continue

        matched = re.match(r"(\S+)\s+(.+)", line)
        if matched is None:
            continue
        heading, pronunciation = matched.groups()
        if len(set(heading) - set(CHARS)) > 0:
            # some characters other than A-Z
            continue
        pronunciation = pronunciation.split(" ")
        accent_index = -1
        phons = []
        for i, phoneme in enumerate(pronunciation):
            stress = "0"
            if phoneme[-1] in "012":
                stress = phoneme[-1]
                phoneme = phoneme[:-1]
            if stress == "1":
                # primary stress
                accent_index = i
            phons.append(phoneme)

        saved_words.append((
            tuple(char2id[c] for c in heading),
            tuple(phon2id[p] for p in phons),
            accent_index
        ))

    print()

    f = open("dataset.pickle", "wb")
    pickle.dump({
        "chars": CHARS,
        "phonemes": PHONEMES,
        "data": saved_words,
        "LICENSE": """Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    The contents of this file are deemed to be source code.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

    This work was supported in part by funding from the Defense Advanced
    Research Projects Agency, the Office of Naval Research and the National
    Science Foundation of the United States of America, and by member
    companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
    the contributions of many volunteers to the expansion and improvement of
    this dictionary.

    THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
    ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
    NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    }, f)
    f.close()