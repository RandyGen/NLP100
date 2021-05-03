import collections

path = "popular-names.txt"

new_list = []
head_list = []

with open(path, "r") as f:
    for line in f:
        new_list.append(line.split())

for word in new_list:
    head_list.append(word[0])

c = collections.Counter(head_list)

print(c.most_common())
