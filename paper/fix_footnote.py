content = open('main_ral.tex', encoding='utf-8').read()

old = r"""trajectory.\footnote{Real neighbours exclude: (i)~the ego agent (slot
0, whose future matches ground truth); (ii)~zero-occupancy padding
slots (all-zero future trajectories). Only non-sentinel, non-ego,
non-zero slots are checked.}"""

new = r"""trajectory (excluding the ego agent, padding slots, and zero-occupancy
sentinels)."""

if old in content:
    content = content.replace(old, new)
    print('Footnote folded into prose: OK')
else:
    print('NOT FOUND - whitespace may differ, check manually')

open('main_ral.tex', 'w', encoding='utf-8').write(content)
print('File written.')
