string = "stressed"

string_list = list(string)
length = len(string_list)

new_string = ''

for hoge in range(length):
    new_string += string_list.pop(-1)

print(new_string)