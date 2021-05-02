path = "popular-names.txt"

with open(path, "r") as f:
    for line in f:
        print(line.replace("\t", " "))
