#!/usr/bin/env python3

import sys
import random

random.seed(42)

for line in sys.stdin:
    try:
        id = line.strip()
    except ValueError as e:
        continue

    print("%d,%s" % (random.randint(0, 4), id))
