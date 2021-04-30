string1 = "paraparaparadise"
string2 = "paragraph"

x = set()
y = set()

def n_gram(n, s):
    l = []
    words = ""
    for i in range(len(s)):
        for j in range(len(s)):
            if i <= j < len(s) and j < i+n:
                words += s[j]
        if i+1 != len(s):
            l.append(words)
        words = ""

    return set(l)

x = n_gram(2, string1)
y = n_gram(2, string2)
print(x)
print(y)

XorY = x | y
print(XorY)

XandY = x & y
print(XandY)

XdisY = x - y
print(XdisY)

if {'se'} <= x:
    print("True")
else:
    print("False")
if {'se'} <= y:
    print("True")
else:
    print("False")
