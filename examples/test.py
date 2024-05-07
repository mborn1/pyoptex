import numpy as np
import re

script = '-1 - `A` + 5'

print(re.sub(r'(?<!Col\()(-*(?=[\.\d]+)[\.\d]+)', lambda m: f'Col({m.group(1)}, (effect_types, col_start), is_constant=True)', script))
