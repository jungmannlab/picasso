"""Sphinx configuration for the Picasso documentation.

See https://www.sphinx-doc.org/en/master/usage/configuration.html.
"""

import os
import pkgutil
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

# -- Project information -----------------------------------------------------

project = "Picasso"
copyright = "2019-2026, Jungmann Lab"
author = "Jungmann Lab"

# Read the version from picasso/version.py rather than importing the
# package, so that the version is known even if an import fails.
_version_globals = {}
with open(os.path.join(_ROOT, "picasso", "version.py")) as _vf:
    exec(_vf.read(), _version_globals)
release = _version_globals["__version__"]
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Markdown (the changelog): generate anchors for headings so that the
# release sections can be linked to.
myst_heading_anchors = 2
myst_enable_extensions = ["colon_fence"]
# The changelog skips heading levels (## release, #### detail).
suppress_warnings = ["myst.header"]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
# Document exactly the names in each module's __all__, in that order
# (modules without __all__ still list all their public members).
autosummary_ignore_module_all = False
# Members are listed by the autosummary module template
# (_templates/autosummary/module.rst), not by automodule.
autodoc_default_options = {"member-order": "bysource"}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# Heavy, optional or hardware-specific imports that are not needed to read
# the docstrings. Mocking them keeps the API pages building on Read the
# Docs, which has no GPU.
autodoc_mock_imports = ["wgpu", "PyImarisWriter", "hdf5plugin"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# Class "Attributes" sections as fields, so that attributes also picked up
# as members are not described twice.
napoleon_use_ivar = True
# Keep the type strings as written; converting them to references
# garbles unions such as ``Callable[[int], None] | None``.
napoleon_preprocess_types = False

# :DOI:`10.xxxx/yyyy` in docstrings (role names are case-insensitive).
extlinks = {"doi": ("https://doi.org/%s", "DOI: %s")}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


_MODULE_TITLE = re.compile(r"^picasso[\w.]*$")


def _strip_module_title(app, what, name, obj, options, lines):
    """Drop the ``picasso.module`` / ``~~~~`` title of module docstrings.

    The title would otherwise become a section inside the generated page,
    which already has the module name as its title.
    """
    if what != "module":
        return
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start + 1 >= len(lines):
        return
    title, underline = lines[start].strip(), lines[start + 1].strip()
    if (
        _MODULE_TITLE.match(title)
        and underline
        and set(underline) <= {"~", "=", "-"}
    ):
        del lines[: start + 2]


#: Where autosummary writes the module pages (not under version control)
_GENERATED = os.path.join(_HERE, "api", "generated")


def _submodules(name: str) -> list[str]:
    """All submodules of the package ``name`` in the repository, found on
    disk (nothing is imported)."""
    package = os.path.join(_ROOT, *name.split("."))
    if not os.path.isfile(os.path.join(package, "__init__.py")):
        return []
    names = []
    for module in pkgutil.iter_modules([package]):
        full_name = f"{name}.{module.name}"
        names.append(full_name)
        if module.ispkg:
            names.extend(_submodules(full_name))
    return names


def _prune_stale_api_pages(app):
    """Delete the module pages in ``api/generated`` that the API index no
    longer lists.

    Autosummary only adds and updates pages, so the page of a module
    taken out of ``api/index.rst`` (or of a submodule of a package no
    longer listed ``:recursive:``) would stay and warn that it is in no
    toctree, documenting its members a second time.
    """
    from sphinx.ext.autosummary.generate import find_autosummary_in_files

    if not os.path.isdir(_GENERATED):
        return
    expected = set()
    for entry in find_autosummary_in_files(
        [os.path.join(_HERE, "api", "index.rst")]
    ):
        if entry.path is None:  # listed, but no page generated
            continue
        expected.add(entry.name)
        if entry.recursive:
            expected.update(_submodules(entry.name))
    for page in os.listdir(_GENERATED):
        name, extension = os.path.splitext(page)
        if extension == ".rst" and name not in expected:
            os.remove(os.path.join(_GENERATED, page))


def setup(app):
    app.connect("autodoc-process-docstring", _strip_module_title)
    # before autosummary generates the pages (priority 500)
    app.connect("builder-inited", _prune_stale_api_pages, priority=400)


# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"Picasso {release}"
html_static_path = ["_static"]
# pages copied as they are, e.g. redirects from renamed pages
html_extra_path = ["_extra"]
html_css_files = ["custom.css"]
html_js_files = ["anchor-redirects.js"]
html_show_sourcelink = False

html_favicon = "_static/picasso-logo.png"

html_theme_options = {
    "logo": {
        "image_light": "_static/picasso-logo.png",
        "image_dark": "_static/picasso-logo-dark.png",
        "alt_text": "Picasso",
    },
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "use_edit_page_button": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jungmannlab/picasso",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/picassosr/",
            "icon": "fa-brands fa-python",
        },
    ],
}

html_context = {
    "github_user": "jungmannlab",
    "github_repo": "picasso",
    "github_version": "master",
    "doc_path": "docs",
}

# Pages without a left sidebar: the landing page has nothing to navigate.
html_sidebars = {"index": [], "changelog": []}

# Development branches (vX.Y) are built as separate Read the Docs versions
# for the test builds of Picasso, whose help buttons link there (see
# picasso.docs_url). Flag them so readers know the page is not the
# released documentation.
_rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
if re.fullmatch(r"v\d+\.\d+", _rtd_version):
    html_theme_options["announcement"] = (
        f"This is the documentation of the <b>test version {release}</b> of "
        "Picasso, with features that are not yet released. See the "
        '<a href="https://picassosr.readthedocs.io/en/latest/">main '
        "documentation</a> for the released version."
    )
