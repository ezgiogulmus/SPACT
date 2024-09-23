import sys

file_path = sys.argv[1]
with open(file_path, 'r') as f:
    lines = f.readlines()

if lines != []:
    print('############################ ERROR ###########################')
    for line in lines:
        print(line)
else:
    print('############################ OK ###########################')