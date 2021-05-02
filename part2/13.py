path1 = "col1.txt"
path2 = "col2.txt"

new_list = []
f1_list = []
f2_list = []

with open(path1, "r") as f1:
    for word in f1:
        f1_list.append("".join(word.rsplit()))

with open(path2, "r") as f2:
    for word in f2:
        f2_list.append("".join(word.rsplit()))

for i, word in enumerate(f1_list):
    new_list.append(word + "\t" + f2_list[i] + "\n")

with open("marged.txt", "w") as f:
    f.write("".join(new_list))
