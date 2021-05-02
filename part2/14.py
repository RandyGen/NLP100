import sys
import fileinput
from pathlib import Path

if Path(sys.argv[1]).exists():
    for i, line in enumerate(fileinput.input()):
        print(line)
        if i+1 == int(sys.argv[2]):
            break
