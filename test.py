import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from collections import namedtuple

A = namedtuple('A', 'Z ratio', defaults=(None, 1))
print(A._field_defaults)
