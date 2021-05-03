path = "popular-names.txt"

new_list = []

with open(path, "r") as f:
    for line in f:
        new_list.append(line.split())

print(new_list[:5])

sorted_list = sorted(new_list, key=lambda student: int(student[2]))

print(sorted_list[:5])
