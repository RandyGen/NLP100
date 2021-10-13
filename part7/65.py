sem_cor = 0
sem_cnt = 0
syn_cor = 0
syn_cnt = 0

with open('data/analogy_data_add.txt', 'r') as f:
    for line in f:
        line = line.split()
        if line[0].startswith('gram'):
            sem_cnt += 1
            if line[4] == line[5]:
                sem_cor += 1
        else:
            syn_cnt += 1
            if line[4] == line[5]:
                syn_cor += 1

print(f'意味的アナロジー正解率: {sem_cor/sem_cnt:.3f}')
print(f'文法的アナロジー正解率: {syn_cor/syn_cnt:.3f}')
