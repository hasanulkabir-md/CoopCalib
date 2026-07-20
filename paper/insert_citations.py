content = open('main_ral.tex', encoding='utf-8').read()

replacements = [
    # A1: Minderer — after Karandikar sentence in §II-A
    (
        'No prior work has reported or optimised ECE for trajectory prediction;\nour V0 baseline closes this gap.',
        'While Minderer~\\etal~\\cite{minderer2021revisiting} showed that modern architectures such as vision transformers exhibit improved calibration over ResNets, no prior work has reported or optimised ECE for trajectory prediction; our V0 baseline closes this gap.'
    ),
    # A2: Nixon — after niculescu sentence in §II-D
    (
        'Reliability diagrams~\\cite{niculescu2005predicting} established the\nvisual diagnostic for calibration that underpins our ECE computation.',
        'Reliability diagrams~\\cite{niculescu2005predicting} established the visual diagnostic for calibration that underpins our ECE computation. Nixon~\\etal~\\cite{nixon2019measuring} proposed complementary scoring rules for calibration measurement; we adopt ECE for consistency with the classification literature.'
    ),
    # B1: Kothari + B_sgcn: SGCN — after Social-STGCNN sentence
    (
        'Social-STGCNN~\\cite{mohamed2020social} used spatial graphs;\nTUTR~\\cite{shi2023tutr} replaced these with full pairwise\nself-attention.',
        'Social-STGCNN~\\cite{mohamed2020social} used spatial graphs; SGCN~\\cite{shi2021sgcn} introduced sparsity into graph convolution for efficiency. Kothari~\\etal~\\cite{kothari2021interpretable} incorporated interpretable rule-based interaction anchors. TUTR~\\cite{shi2023tutr} replaced graph convolution with full pairwise self-attention.'
    ),
    # B2: AgentFormer — after TUTR sentence
    (
        'Recent diffusion-based~\\cite{liu2024diftraj} and\ngeometric~\\cite{wong2024socialcircle} approaches further advance\ninteraction modelling~\\cite{bae2024singular}, but none report\ncalibration, FPR, SVR, or SPSR.',
        'Concurrent transformer approaches include AgentFormer~\\cite{yuan2021agentformer}, which uses agent-aware attention for socio-temporal forecasting. Recent diffusion-based~\\cite{liu2024diftraj} and geometric~\\cite{wong2024socialcircle} approaches further advance interaction modelling~\\cite{bae2024singular}, but none report calibration, FPR, SVR, or SPSR.'
    ),
    # D1: Robicquet + D2: Ziebart + D4: Sun — in §II-D after Trajectron++ sentence
    (
        'Trajectron++~\\cite{salzmann2020trajectron} evaluated prediction\nquality under structured noise, but did not report ECE or social\ncompliance.',
        'Trajectron++~\\cite{salzmann2020trajectron} evaluated prediction quality under structured noise, but did not report ECE or social compliance. The SDD dataset~\\cite{mangalam2020pecnet} was introduced alongside social etiquette metrics~\\cite{robicquet2016learning} that anticipate our SVR formulation. Planning-based prediction~\\cite{ziebart2009planning} established that pedestrian intent should inform trajectory forecasting, an assumption our SPSR metric operationalises. The relationship between prediction quality and planning success has been formalised in recent work~\\cite{sun2022complementing}, motivating our SPSR metric.'
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
print(f'Done. {count}/5 blocks replaced.')
