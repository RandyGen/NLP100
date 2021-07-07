class Morphs:
    def __init__(self, dc):
        self.surface = dc['surface']
        self.base = dc['base']
        self.pos = dc['pos']
        self.pos1 = dc['pos1']


if __name__ == '__main__':
    c = 0
    fname = './ai.ja.text.parsed'
    with open(fname, 'r') as f:
        for line in f:
            if line == 'EOS\n' or line[0] == '*' or line == '\n':
                continue
            else:
                (surface, attr) = line.split('\t', 1)
                attr = attr.split(',')
                line_dict = {
                    'surface': surface,
                    'base': attr[5],
                    'pos': attr[0],
                    'pos1': attr[1]
                }
                if c < 5:
                    morphs = Morphs(line_dict)
                    print(vars(morphs))
                    c += 1
