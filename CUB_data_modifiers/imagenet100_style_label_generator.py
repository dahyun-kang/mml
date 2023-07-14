import os

if __name__ == '__main__':
    input_file = './classes.txt'
    output_file = './labels.txt'

    with open(input_file, "r") as rf:
        data = rf.readlines()

    _lines = [l.split() for l in data]
    _new_lines = [f"{l[1]} {int(l[0])-1} {l[1].split('.')[1]}\n" for l in _lines]

    with open(output_file, "w") as wf:
        wf.writelines(_new_lines)
