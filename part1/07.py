x=12
y="気温"
z=22.4

def templete(a, b, c):
    return f"{a}時の{b}は{c}"


new_string = templete(x, y, z)
print(new_string)
    