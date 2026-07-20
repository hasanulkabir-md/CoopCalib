content = open('main_ral.tex', encoding='utf-8').read()
replacements = [
    (r'\subsection{Calibration Loss \texorpdfstring{$\mathcal{L}_{\text{ECE}}$}{L\_ECE}}',
     r'\subsection{Calibration Loss $\mathcal{L}_{\text{ECE}}$}'),
    (r'\subsection{Social Energy Loss \texorpdfstring{$\mathcal{L}_{\text{energy}}$}{L\_energy}}',
     r'\subsection{Social Energy Loss $\mathcal{L}_{\text{energy}}$}'),
    (r'\subsection{Ranking-Aware Social Loss \texorpdfstring{$\mathcal{L}_{\text{ranking}}$}{L\_ranking}}',
     r'\subsection{Ranking-Aware Social Loss $\mathcal{L}_{\text{ranking}}$}'),
    (r'\subsection{Dynamic KL Prior \texorpdfstring{$\mathcal{L}_{\text{KL}}$}{L\_KL}}',
     r'\subsection{Dynamic KL Prior $\mathcal{L}_{\text{KL}}$}'),
]
count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        count += 1
        print(f'OK: {old[:50]}')
    else:
        print(f'MISS: {old[:50]}')
open('main_ral.tex', 'w', encoding='utf-8').write(content)
print(f'Done. {count}/4 replaced.')
