string = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

new_list = []

string = string.replace(",", "").replace(".", "")

for word in string.split():
    new_list.append(len(word))

print(new_list)