# unsolved

import sys
import fileinput
from pathlib import Path


with open(sys.argv[1], "r", encoding="utf-8") as f:
    i = 0
    for line in reversed(f.splitlines()):
        print(line)
        i += 1
        if i == int(sys.argv[2]):
            break
