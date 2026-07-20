with open('ref.bib', encoding='utf-8') as f:
    content = f.read()

import re
entries = re.split(r'\n(?=@)', content.strip())
seen = {}
duplicates = []
unique = []
for e in entries:
    m = re.match(r'@\w+\{(\S+),', e.strip())
    if m:
        key = m.group(1).rstrip(',')
        if key in seen:
            duplicates.append(key)
        else:
            seen[key] = True
            unique.append(e)
    else:
        unique.append(e)

print(f'Total entries: {len(entries)}')
print(f'Unique entries: {len(unique)}')
print(f'Duplicates removed: {duplicates}')

with open('ref.bib', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(e.strip() for e in unique))
print('ref.bib cleaned.')
