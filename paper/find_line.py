lines = open('main_ral.tex', encoding='utf-8').readlines()
for i, l in enumerate(lines, 1):
    if 'architectural fix' in l and 'karandikar' in l:
        print(f'L{i}: {repr(l)}')
