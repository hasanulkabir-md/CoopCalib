content = open('main_ral.tex', encoding='utf-8').read()

old = 'Sampling-based training objectives---where all $ candidates receive\ngradient proportional to their predicted weight---are the natural\narchitectural fix~\\cite{karandikar2021soft}.\n\n\\paragraph{Why FPR is identically zero}'

new = 'Sampling-based training objectives---where all $ candidates receive gradient proportional to their predicted weight---are the natural architectural fix~\\cite{karandikar2021soft}. Motion Transformer~\\cite{shi2023mtr} exemplifies this approach at scale, achieving state-of-the-art results on Waymo by training all motion modes jointly.\n\n\\paragraph{Why FPR is identically zero}'

if old in content:
    content = content.replace(old, new)
    print('OK: MTR citation inserted.')
else:
    print('MISS: still not matching.')

open('main_ral.tex', 'w', encoding='utf-8').write(content)
