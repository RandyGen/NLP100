class Morphs:
    def __init__(self, morphs):
        (surface, attr) = morphs.split('\t', 1)
        attr = attr.split(',')
        self.surface = surface
        self.base = attr[5]
        self.pos = attr[0]
        self.pos1 = attr[1]


if __name__ == '__main__':
    c = 0
    sentence = []
    morphs = []
    fname = './ai.ja.text.parsed'

    with open(fname, 'r') as f:
        for line in f:
            if line[0] == '*':
                continue
            elif line != 'EOS\n':
                morphs.append(Morphs(line))
            else:
                sentence.append(morphs)
                morphs = []

    for m in sentence[2]:
        print(vars(m))
