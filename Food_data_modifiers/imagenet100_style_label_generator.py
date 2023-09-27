import os

if __name__ == '__main__':
    input_file = './meta/classes.txt'
    output_file = './labels.txt'

    with open(input_file, "r") as rf:
        data = rf.readlines()

    _lines = [l.split() for l in data]
    _new_lines = [f"{l[0]} {idx} {l[0]}\n" for idx, l in enumerate(_lines)]

    with open(output_file, "w") as wf:
        wf.writelines(_new_lines)
