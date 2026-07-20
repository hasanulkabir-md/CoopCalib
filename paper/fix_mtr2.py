lines = open('main_ral.tex', encoding='utf-8').readlines()
old = 'architectural fix~\\cite{karandikar2021soft}.\n'
new = 'architectural fix~\\cite{karandikar2021soft}. Motion Transformer~\\cite{shi2023mtr} exemplifies this approach at scale, achieving state-of-the-art results on Waymo by training all motion modes jointly.\n'
if lines[695] == old:
    lines[695] = new
    print('OK: MTR citation inserted at L696.')
else:
    print(f'MISS: L696 contains: {repr(lines[695])}')
open('main_ral.tex', 'w', encoding='utf-8').write(''.join(lines))
