content = open('main_ral.tex', encoding='utf-8').read()

replacements = [
    # Rudenko survey — add at start of §II-D after "Standard trajectory evaluation" sentence
    (
        'Standard trajectory evaluation relies on minADE and minFDE, metrics\nthat measure geometric accuracy but ignore probability calibration\nand social compliance~\\cite{alahi2016social,gupta2018social}.',
        'Standard trajectory evaluation relies on minADE and minFDE, metrics that measure geometric accuracy but ignore probability calibration and social compliance~\\cite{alahi2016social,gupta2018social}. Rudenko~\\etal~\\cite{rudenko2020human} survey the field and identify calibration and social compliance as open challenges unaddressed by displacement metrics.'
    ),
    # PRECOG — add after ziebart sentence in §II-D
    (
        'Planning-based prediction~\\cite{ziebart2009planning} established that pedestrian intent should inform trajectory forecasting, an assumption our SPSR metric operationalises.',
        'Planning-based prediction~\\cite{ziebart2009planning} established that pedestrian intent should inform trajectory forecasting, an assumption our SPSR metric operationalises. PRECOG~\\cite{rhinehart2019precog} extended this to goal-conditioned multi-agent prediction, explicitly linking predicted trajectories to planning outcomes as we do via SPSR.'
    ),
]

count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f'OK: {old[:55].strip()}...')
        count += 1
    else:
        print(f'MISS: {old[:55].strip()}...')

open('main_ral.tex', 'w', encoding='utf-8').write(content)
print(f'Done. {count}/2 replaced.')
