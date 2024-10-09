import setuptools
import site

site.ENABLE_USER_SITE = 1

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyoptex",
    version="0.0.1",
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
            'sphinx~=4.4',
            'docutils<0.18',
            'numpydoc~=1.2',
            'pydata_sphinx_theme~=0.7',
            'sphinx-copybutton~=0.5'
        ],
        'examples': [
            'openpyxl==3.0.10'
        ]
    }
)