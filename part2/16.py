import sys

# list = []

with open(sys.argv[1], "r") as f:
    
    for i, line in enumerate(f):
        print(line)
        # list.append(line)

        if i % int(sys.argv[2]) == 0:
            print("##########")
            # with open(path, "w") as f:
            #     f.write("".join(list))
            # list = []    
