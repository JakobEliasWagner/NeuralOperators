import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

# -- Project information -----------------------------------------------------
project = "Neural Operators"
copyright = "2023, Jakob Wagner"
author = "Jakob Wagner"

# -- General configuration ---------------------------------------------------
extensions = ["sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]
