string = "I am an NLPer"

new_list = []

def n_gram(n, s, basis):
    l = []

    if basis == "単語":
        words = []
        for i in range(len(s.split())):
            for j, word in enumerate(s.split()):
                if i <= j < len(s.split()) and j < i+n:
                    words.append(word)
            if i+1 != len(s.split()):
                l.append(' '.join(words))
            words = []
    
    elif basis == "文字":
        words = ""
        s = s.replace(" ", "")
        for i in range(len(s)):
            for j in range(len(s)):
                if i <= j < len(s) and j < i+n:
                    words += s[j]
            if i+1 != len(s):
                l.append(words)
            words = ""

    return l


new_list = n_gram(2, string, "単語")
print(new_list)

new_list = n_gram(2, string, "文字")
print(new_list)
