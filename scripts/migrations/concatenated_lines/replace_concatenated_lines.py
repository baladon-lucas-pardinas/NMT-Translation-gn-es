import sys

"""
There were 2 problems during the model test inference:
1. The test set had 10 lines which were concatenated in positions:
    - 1000
    - 1999
    - 2998
    - 3997
    - 4997
2. The last lines was missing
"""

def fix_file(file_dir: str, fixed_lines_dir: str):
    """
    Input: 
        - The wrong file
        - A file where lines from 0-9 
        are the lines missing in the mentioned positions

    Output: 
        - The fixed file
    """

    to_fix_lines = [999, 1998, 2997, 3996, 4996]

    with open(file_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(fixed_lines_dir, 'r', encoding='utf-8') as f:
        fixed_lines = f.readlines()

    new_lines = []
    for i, line in enumerate(lines):
        if i in to_fix_lines:
            new_lines.append(fixed_lines.pop(0))
            new_lines.append(fixed_lines.pop(0))
        else:
            new_lines.append(line)


    if len(fixed_lines) == 1:
        new_lines.append(fixed_lines.pop(0))

    assert len(fixed_lines) == 0, "Not all lines were used"

    with open(file_dir, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
if __name__ == '__main__':
    file_dir = sys.argv[1]
    fixed_lines_dir = sys.argv[2]
    fix_file(file_dir, fixed_lines_dir)