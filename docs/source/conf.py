# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# Include paths to modules so autodoc can grab the docstrings and signatures
import sys
sys.path.insert(0,'../evoke/src')           # for local build
sys.path.insert(0,'../../evoke/src')        # for ReadTheDocs build
sys.path.insert(0,'../evoke/examples')      # for local build
sys.path.insert(0,'../../evoke/examples')   # for ReadTheDocs build

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'evoke'
copyright = '2023, Manolo Martínez & Stephen Mann'
author = 'Manolo Martínez & Stephen Mann'
version = '0.1'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

templates_path = ['_templates']
exclude_patterns = []

root_doc = 'index'

# Order classes and methods by the same order as in the source files
autodoc_member_order = 'bysource'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
