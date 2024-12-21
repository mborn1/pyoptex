import re
import setuptools

with open("src/pyoptex/__init__.py", "r", encoding="utf-8") as f:
    version = re.search(r'__version__ = [\'"](.*)[\'"]', f.read()).group(1)

setuptools.setup(
    version=version,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
