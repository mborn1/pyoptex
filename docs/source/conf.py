# Configuration file for the Sphinx documentation builder.

import datetime
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parents[2].resolve() / 'src'))
from pyoptex import __version__ as lib_version

# -- Project information

project = 'PyOptEx'
copyright = '2024, Mathias Born'
author = 'Mathias Born'

release = lib_version
version = lib_version.split('-')[0]

rst_epilog = f"""
.. |version| replace:: {version}

.. |release| replace:: {release}

.. |date| replace:: {format(datetime.datetime.now(), '%B %d, %Y')}
"""

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None)
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

# -- Options for EPUB output
epub_show_urls = 'footnote'
