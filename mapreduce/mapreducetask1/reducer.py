#!/usr/bin/env python3

import sys
import random

random.seed(42)

ids_in_line = random.randint(1, 5)
ids_list = []

for line in sys.stdin:
    try:
        val, id = line.strip().split(',', 1)
    except ValueError as e:
        continue

    if ids_in_line == 0:
        print(",".join(ids_list))

        # reset values
        ids_in_line = random.randint(1, 5)
        ids_list = []
    else:
        ids_list.append(id)
        ids_in_line -= 1

if ids_list:
    print(",".join(ids_list))
