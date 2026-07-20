import re

with open('main_ral.tex', encoding='utf-8') as f:
    content = f.read()

fixes = [
    # Fix 1: remove raggedbottom
    (r'\\raggedbottom\n', ''),
    # Fix 2: reduce totalnumber from 6 to 4
    (r'\\setcounter\{totalnumber\}\{6\}', r'\\setcounter{totalnumber}{4}'),
    # Fix 3: reduce topnumber from 4 to 3
    (r'\\setcounter\{topnumber\}\{4\}', r'\\setcounter{topnumber}{3}'),
    (r'\\setcounter\{dbltopnumber\}\{4\}', r'\\setcounter{dbltopnumber}{3}'),
]

for pattern, replacement in fixes:
    before = content
    content = re.sub(pattern, replacement, content)
    found = 'OK' if content != before else 'NOT FOUND - check manually'
    print(f'{pattern[:40]} -> {found}')

with open('main_ral.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done. File written.')
