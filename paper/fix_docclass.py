content = open('main_ral.tex', encoding='utf-8').read()

# Fix 1: remove letterpaper from documentclass
old1 = r'\documentclass[journal,letterpaper]{IEEEtran}'
new1 = r'\documentclass[journal]{IEEEtran}'

# Fix 2: move setcounters into preamble (before \begin{document})
# Remove them from after \begin{document}
old2 = r"""\begin{document}
% Float patience — prevents float buildup causing vbox badness
\setcounter{topnumber}{3}
\setcounter{dbltopnumber}{3}
\setcounter{totalnumber}{4}"""

new2 = r"""\setcounter{topnumber}{3}
\setcounter{dbltopnumber}{3}
\setcounter{totalnumber}{4}

\begin{document}"""

count = 0
for old, new, label in [(old1,new1,'letterpaper'),(old2,new2,'setcounters')]:
    if old in content:
        content = content.replace(old, new)
        print(f'OK: {label}')
        count += 1
    else:
        print(f'MISS: {label}')

open('main_ral.tex', 'w', encoding='utf-8').write(content)
print(f'Done. {count}/2 fixed.')
