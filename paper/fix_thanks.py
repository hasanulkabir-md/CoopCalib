
lines = open('main_ral.tex', encoding='utf-8').readlines()

old = '\\thanks{Manuscript received April 14, 2026.\nThis work was supported by [funding source].\n(Corresponding author: Author One.)\nAuthor One, Author Two, and Author Three are with\n[Institution], [City], [Country]\n(e-mail: author@institution.edu.cn).}}'

new = '\\thanks{Manuscript received. This work received no funding. (Corresponding author: Author One.) Author One, Author Two, and Author Three are with [Institution], [City], [Country] (e-mail: author@institution.edu.cn).}}'

content = ''.join(lines)

if old in content:

    content = content.replace(old, new)

    print('OK: thanks block updated.')

else:

    print('MISS')

open('main_ral.tex', 'w', encoding='utf-8').write(content)

