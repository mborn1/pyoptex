import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from collections import namedtuple

__A__ = namedtuple('__A__', 'Z ratio', defaults=(None, 1))
class A(__A__):
    def __eq__(self, other):
        Zeq = np.all(self.Z == other.Z)
        ratioeq = False
        if isinstance(self.ratio, (tuple, list, np.ndarray)):
            if isinstance(other.ratio, (tuple, list, np.ndarray)):
                ratioeq = np.all(np.array(self.ratio) == np.array(other.ratio))
        else:
            if not isinstance(other.ratio, (tuple, list, np.ndarray)):
                ratioeq = self.ratio == other.ratio
        return Zeq and ratioeq
    
a1 = A(np.array([1, 2]), 2)
a2 = [A(np.array([1, 2]), 2)]
print(a2.index(a1))
