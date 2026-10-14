"""
Classical designs: factorial, screening, response surface, Taguchi,
space-filling and mixture designs, and their evaluation.

Every function accepts the factors of any pyoptex module. The exported
``Factor`` is the generic :py:class:`pyoptex.utils.Factor`. To compare a
classical design with an optimal one, create the factors with the
optimizer's own class instead (e.g.
:py:class:`pyoptex.doe.fixed_structure.Factor`), since the optimizers only
accept their own factor class; the classical functions accept it as well.
"""

# ruff: noqa: F401
from ...utils.factor import Factor
from .box_behnken import box_behnken
from .central_composite import central_composite
from .definitive_screening import definitive_screening
from .doehlert import doehlert
from .factorial import full_factorial
from .fractional_factorial import fractional_factorial, min_aberration_generators
from .latin_hypercube import latin_hypercube
from .mixture import simplex_centroid, simplex_lattice
from .plackett_burman import plackett_burman
from .space_filling import halton, sobol, uniform
from .supersaturated import supersaturated
from .taguchi import taguchi
from .utils import randomize
