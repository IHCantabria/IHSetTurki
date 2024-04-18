# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# from IHSetTurki.__init__ import __version__

__version__ = "0.1.0"
project = "IHSetTurki"
copyright = "2024, Lim, Changbin"
author = "Lim, Changbin"
version = release = __version__

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "ihcantabria",  # Username
    "github_repo": "IHSetTurki",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "myst_nb",
    # 'sphinxcontrib.autodoc_pydantic',
    "sphinx.ext.autosummary",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "display_version": True,
    "style_external_links": False,
}
# html_theme = 'furo'
# html_theme = 'sphinx_book_theme'

html_static_path = ["_static"]
html_logo = ""
html_title = " v" + release
