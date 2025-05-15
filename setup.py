import re
import sys
import os
import pathlib
import setuptools
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

with open("src/pyoptex/__init__.py", "r", encoding="utf-8") as f:
    version = re.search(r'__version__ = [\'"](.*)[\'"]', f.read()).group(1)

# Check if windows
def is_platform_windows():
    return sys.platform in ("win32", "cygwin")

# Check if mac
def is_platform_mac():
    return sys.platform == "darwin"

# Function to search for all .pyx files in the src directory
def search_pyx(root):
    pyx_files = []
    for file in os.listdir(root):
        if file.endswith(".pyx"):
            pyx_files.append(root / file)
        elif os.path.isdir(root / file):
            pyx_files.extend(search_pyx(root / file))
    return pyx_files


# Define extra compile and link arguments
extra_compile_args = []
extra_link_args = []
if is_platform_windows():
    # extra_compile_args.append("/O2")
    pass
else:
    extra_compile_args.append("-O3")

# Define Cython extensions
print( [
    (path.relative_to(pathlib.Path('src')).as_posix().replace('/', '.')[:-4],
        [str(path)]) for path in search_pyx(pathlib.Path('src'))])
extensions = [
    Extension(
        path.relative_to(pathlib.Path('src')).as_posix().replace('/', '.')[:-4],
        [str(path)],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
        extra_compile_args=extra_compile_args, 
        extra_link_args=extra_link_args
    ) for path in search_pyx(pathlib.Path('src'))]

setuptools.setup(
    version=version,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    setup_requires=['cython', 'numpy'],
)
