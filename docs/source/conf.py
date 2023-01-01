# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'In-Vivo-Imaging-Pipeline'
copyright = "2023, Darik A. O'Neil"
author = "Darik A. O'Neil"
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



# Add all modules

import os
import sys
module_names = (
    "ExperimentManagement",
    "Behavior",
    "Imaging",
    "Theory"
)

for _name in module_names:
    _path = "".join(["..", "\\", _name])
    sys.path.insert(0, os.path.abspath(_path))
