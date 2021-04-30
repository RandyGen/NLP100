string1 = "パトカー"
string2 = "タクシー"

new_string = ""

string_list1 = list(string1)
string_list2 = list(string2)

for i in range(len(string_list1)):
    new_string += string_list1.pop(0)
    new_string += string_list2.pop(0)

print(new_string)
