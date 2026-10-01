#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : conf.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>


import  os
import  sys
import  runpy
from    pathlib  import  Path


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
os.environ["MPLBACKEND"] = "Agg"

project = "ACCES"
copyright = "2026, the Coexist developers"
release = runpy.run_path(root / "coexist/__version__.py")["__version__"]
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]
root_doc = "index"
autosummary_generate = True
numpydoc_show_class_members = False
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": (
        "https://github.com/uob-positron-imaging-centre/ACCES-CoExSiST"
    ),
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "uob-positron-imaging-centre",
    "github_repo": "ACCES-CoExSiST",
    "github_version": "main",
    "doc_path": "docs/source",
}
