import re
content = open('main_ral.tex', encoding='utf-8').read()
before = content.count('texorpdfstring')
content = re.sub(r'\\texorpdfstring\{([^}]*)\}\{[^}]*\}', r'\1', content)
after = content.count('texorpdfstring')
print(f'Replaced {before - after} of {before} occurrences')
open('main_ral.tex', 'w', encoding='utf-8').write(content)
print('File written.')
