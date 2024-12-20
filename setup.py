import site
import re
import setuptools

site.ENABLE_USER_SITE = 1

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("src/pyoptex/__init__.py", "r", encoding="utf-8") as f:
    version = re.search(r'__version__ = [\'"](.*)[\'"]', f.read()).group(1)

setuptools.setup(
    name="pyoptex",
    version=version,
    author="Mathias Born",
    author_email="mathiasborn2@gmail.be",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        'numba==0.55.1',
        'numpy==1.21.5',
        'tqdm==4.64.0',
        'scipy==1.8.0',
        'pandas==1.5.3',
        'plotly==5.22.0',
    ],
    extras_require={
        'dev': [
            'sphinx==7.1.2',
            'sphinx-rtd-theme==1.3.0rc1',
            'sphinx-copybutton==0.5.2'
        ],
        'examples': [
            'openpyxl==3.0.10'
        ]
    }
)