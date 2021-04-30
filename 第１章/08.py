import re

string = "Hello World 2021!"

new_string = ""

def cipher(s):
    string = ""
    for i in range(len(s)):
        if re.fullmatch('[a-z]', s[i]):
            string += chr(219 - ord(s[i]))
        else:
            string += s[i]
    return string

# encoder
new_string = cipher(string)
print(new_string)

#decoder
new_string = cipher(new_string)
print(new_string)