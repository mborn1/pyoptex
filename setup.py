import re
import setuptools
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

with open("src/pyoptex/__init__.py", "r", encoding="utf-8") as f:
    version = re.search(r'__version__ = [\'"](.*)[\'"]', f.read()).group(1)

# Define Cython extensions
extensions = [
    Extension(
        "pyoptex.doe.fixed_structure._optimize_cy",
        ["src/pyoptex/doe/fixed_structure/_optimize_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
        # extra_compile_args=["-O3"], 
    ),
    Extension(
        "pyoptex.utils._design_cy",
        ["src/pyoptex/utils/_design_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c"
    ),
    Extension(
        "pyoptex.doe.fixed_structure._init_cy",
        ["src/pyoptex/doe/fixed_structure/_init_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c"
    ),
    Extension(
        "pyoptex._seed_cy",
        ["src/pyoptex/_seed_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c"
    )
]

setuptools.setup(
    version=version,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    setup_requires=['cython', 'numpy'],
)
