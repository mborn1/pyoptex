import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from collections import namedtuple

a = pd.DataFrame({'A': [1, 4, 5], 'C': [0, 1, 2]})
print(a)

print(a.loc[np.array([True, False, True]), ['A']])


