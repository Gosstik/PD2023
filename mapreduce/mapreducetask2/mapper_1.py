#!/usr/bin/env python3

import sys
import re

for line in sys.stdin:
    try:
        line = line.strip()
    except ValueError as e:
        continue

    line = re.sub("[^A-Za-z\\s]", "", line)
    words = re.split("\s+", line)

    for word in words:
        word = word.lower()
        if len(word) < 3:
            continue

        print("%s\t%s\t1" % (''.join(sorted(word)), word))
