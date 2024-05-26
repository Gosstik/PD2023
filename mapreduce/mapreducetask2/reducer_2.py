#!/usr/bin/env python3

import sys

for line in sys.stdin:
    try:
        count, perm, words = line.strip().split('\t', 2)
    except ValueError as e:
        continue

    print("{}\t{}\t{}".format(perm, count, words))
