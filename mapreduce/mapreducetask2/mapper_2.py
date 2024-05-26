#!/usr/bin/env python3

import sys

for line in sys.stdin:
    try:
        perm, count, words = line.strip().split('\t', 2)
    except ValueError as e:
        continue

    print("{}\t{}\t{}".format(count, perm, words))
