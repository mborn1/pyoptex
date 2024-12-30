# Configuration file for the Sphinx documentation builder.

import datetime
import pathlib
import sys
import inspect
import importlib
from urllib.parse import quote

sys.path.append(str(pathlib.Path(__file__).parents[2].resolve() / 'src'))
from pyoptex import __version__ as lib_version

# -- Project information

project = 'PyOptEx'
copyright = '2024, Mathias Born'
author = 'Mathias Born'

release = lib_version
version = lib_version

rst_epilog = f"""
.. |version| replace:: {version}

.. |release| replace:: {release}

.. |date| replace:: {format(datetime.datetime.now(), '%B %d, %Y')}

.. |br| raw:: html

     <br>
"""

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
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


def linkcode_resolve(domain, info):
    """
    Resolves the url for the source code.
    """
    # Only consider Python files
    if domain != 'py':
        return None
    
    # Retrieve start and end line
    start_line, end_line = None, None
    mod = importlib.import_module(info["module"])
    try:
        # Retrieve the object dynamically
        if "." in info["fullname"]:
            objname, attrname = info["fullname"].split(".")
            obj = getattr(mod, objname)
            obj = getattr(obj, attrname)
        else:
            obj = getattr(mod, info["fullname"])

        try:
            # Try to get the source code
            lines, start_line = inspect.getsourcelines(obj)
            end_line = start_line + len(lines)
        except TypeError:
            pass

    except (AttributeError, OSError):
        print(f'Unable to load: {info}')

    # Create URL-encoded filename
    filename = quote(str(info['module']).replace('.', '/'))

    # Create the anchor
    if start_line is None:
        anchor = ''
    else:
        anchor = f'#L{start_line}-L{end_line}'

    # Link to github
    result = "https://github.com/mborn1/pyoptex/blob/%s/src/%s.py%s" % (release, filename, anchor)
    return result
