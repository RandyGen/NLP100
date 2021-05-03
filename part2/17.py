path = "marged.txt"

new_set = set()

with open(path, "r") as f:
    for line in f:
        word = line.split()
        new_set.add(word[0])

print(new_set)
print(len(new_set))