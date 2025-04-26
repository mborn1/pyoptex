import re
import setuptools
from setuptools import Extension
from Cython.Build import cythonize
import numpy

with open("src/pyoptex/__init__.py", "r", encoding="utf-8") as f:
    version = re.search(r'__version__ = [\'"](.*)[\'"]', f.read()).group(1)

# Define Cython extensions
extensions = [
    Extension(
        "pyoptex.doe.fixed_structure._optimize_cy",
        ["src/pyoptex/doe/fixed_structure/_optimize_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setuptools.setup(
    version=version,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    setup_requires=['cython', 'numpy'],
)
