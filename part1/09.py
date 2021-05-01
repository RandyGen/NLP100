import random
string = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

new_string = ""

l = []

for word in string.split():
    if len(word) < 4:
        new_string += word
    else:
        for i in range(len(word)):
            if i != 0 and i+1 != len(word):
                l.append(word[i])
        s = ''.join(random.sample(l, len(l)))
        new_string += word[0] + s + word[len(word)-1]
        l = []
    new_string += ' '

print(new_string)
