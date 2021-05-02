path = "popular-names.txt"

row_counts = 0

with open(path, "r") as f:
    for line in f:
        row_counts += 1

print(f"Number of lines is {row_counts}")
