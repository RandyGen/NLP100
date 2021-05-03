import sys

new_list = []

with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        new_list.append(line)


i = 0
for line in reversed(new_list):
    print(line)
    i += 1
    if i == int(sys.argv[2]):
        break
