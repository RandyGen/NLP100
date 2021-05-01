string = "パタトクカシーー"

new_string1 = ""
new_string2 = ""

for i in range(len(string)):
    if i % 2 == 0:
        new_string1 += string[i]
    else:
        new_string2 += string[i]

print(new_string1)
print(new_string2)
