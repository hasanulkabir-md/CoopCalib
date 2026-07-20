content = open('main_ral.tex', encoding='utf-8').read()

replacements = [
    (
        'Sampling-based training objectives---where all $ candidates receive\ngradient proportional to their predicted weight---are the natural\narchitectural fix~\\cite{karandikar2021soft}.',
        'Sampling-based training objectives---where all $ candidates receive gradient proportional to their predicted weight---are the natural architectural fix~\\cite{karandikar2021soft}. Motion Transformer~\\cite{shi2023mtr} exemplifies this approach at scale, achieving state-of-the-art results on Waymo by training all motion modes jointly.'
    ),
    (
        'Full SVR reduction in dense scenes requires architectural intervention\nat candidate generation, not loss-level pressure.',
        'Full SVR reduction in dense scenes requires architectural intervention at candidate generation, not loss-level pressure. Diffusion-based approaches~\\cite{gu2023motiondiffuser} offer a promising direction by generating diverse candidates through iterative refinement rather than single-pass decoding.'
    ),
]

count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f'OK: {old[:60].strip()}...')
        count += 1
    else:
        print(f'MISS: {old[:60].strip()}...')

open('main_ral.tex', 'w', encoding='utf-8').write(content)
print(f'Done. {count}/2 blocks replaced.')
