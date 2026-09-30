"""Template for creating a Picasso plugin. Save your plugin as a .py file in
~/.picasso/plugins/ (use Plugins -> Open plugins folder in any Picasso app
to find it).

A plugin can extend Picasso in three independent ways, and may use any
combination of them:

* a ``Plugin`` class          -> adds entries to a Picasso GUI's Plugins menu
* a ``PICASSO_API`` mapping   -> names importable from scripts and notebooks
* a ``register_cli`` function -> adds ``picasso <command>`` subcommands

This template shows all three; delete the parts you do not need. A plugin
that only extends the API and the command line needs no Plugin class and no
Qt at all - just delete that section.

For more details, see https://picassosr.readthedocs.io/en/latest/plugins.html

Author:
Date:
"""

# Space to import packages
import numpy as np
from PyQt6 import QtWidgets


def circle_area(radius):
    """Plain function, usable from the GUI, the CLI and API alike."""
    return np.pi * radius**2


# ---------------------------------------------------------------------------
# GUI: adds an entry to the Plugins menu of one Picasso app
# ---------------------------------------------------------------------------


# class that defines modifications to the GUI and actions
class Plugin:
    """Adds an entry to the Plugins menu of one Picasso app.

    Parameters
    ----------
    window : QtWidgets.QMainWindow
        Main window of the Picasso app that loads the plugin.

    Attributes
    ----------
    name : str
        Name of the app the plugin extends, e.g. ``"render"``.
    window : QtWidgets.QMainWindow
        Main window of the Picasso app.
    """

    def __init__(self, window):
        self.name = "render"  # input the name of the app
        self.window = window

    def execute(self):
        """This function is called when opening a GUI."""
        your_action = self.window.plugin_menu.addAction(
            "What does your plugin do?"
        )
        your_action.triggered.connect(self.run_your_plugin)

    def run_your_plugin(self):
        """This function is called when clicking the menu entry.

        Ask the user for a radius and show the area. Note that the work
        itself is done by circle_area, the same function the command line
        and the Python API below use - the GUI only collects the input and
        displays the result.
        """
        radius, ok = QtWidgets.QInputDialog.getDouble(
            self.window,  # parent, so the dialog belongs to the Picasso app
            "Circle area",
            "Radius:",
            value=1.0,
            min=0.0,
            decimals=3,
        )
        if not ok:  # the user pressed Cancel
            return
        QtWidgets.QMessageBox.information(
            self.window,
            "Circle area",
            f"A circle of radius {radius} has an area of "
            f"{circle_area(radius):.3f}.",
        )


# ---------------------------------------------------------------------------
# Python API: names reachable as picasso.plugins.api.<name> from any script
# ---------------------------------------------------------------------------

# Maps the name a script will use to the object you provide. May also be a
# function returning such a dict, if you want to build it lazily.
#
#     from picasso import plugins
#     plugins.api.circle_area(2.0)
#
PICASSO_API = {"circle_area": circle_area}


# ---------------------------------------------------------------------------
# Command line: adds "picasso <command>" subcommands
# ---------------------------------------------------------------------------


def _run_circle_area(args):
    """Handler for the subcommand; receives the parsed arguments."""
    print(
        f"Circle area of radius {args.radius:.3f} is "
        f"{circle_area(args.radius):.3f}"
    )


def register_cli(subparsers):
    """Called with the subparsers of the ``picasso`` command line parser.

    Add one parser per subcommand and name its handler with
    ``set_defaults(func=...)``; the handler is called with the parsed
    ``argparse.Namespace``. Command names must not clash with an existing
    ``picasso`` command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers of the ``picasso`` command line parser.
    """
    parser = subparsers.add_parser(
        "circle-area", help="compute the area of a circle"
    )
    parser.add_argument("radius", type=float)
    parser.set_defaults(func=_run_circle_area)
