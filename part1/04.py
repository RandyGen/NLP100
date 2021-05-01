string = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

new_dict = {}

for i, word in enumerate(string.split()):
    if i+1 == 1 or i+1 == 5 or i+1 == 6 or i+1 == 7 or i+1 == 8 or i+1 == 9 or i+1 == 15 or i+1 == 16 or i+1 == 19:
        new_dict[word[0]] = i+1
    else:
        new_dict[word[:2]] = i+1

print(new_dict)