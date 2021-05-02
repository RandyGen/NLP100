path = "popular-names.txt"

col1 = []
col2 = []

with open(path, "r") as f:
    for line in f:
        for i, word in enumerate(line.split("\t")):
            if i == 0:
                col1.append(word)
            elif i == 1:
                col2.append(word)

with open("col1.txt", "w") as f:
    f.write('\n'.join(col1))

with open("col2.txt", "w") as f:
    f.write('\n'.join(col2))