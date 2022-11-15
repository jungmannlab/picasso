"""
Template for creating a Picasso plugin. Any plugin should be moved to
picasso/picasso/gui/plugins/
Author:
Date:
"""

# Space to import packages
import numpy as np

# Do not change the part below unless stated otherwise
class Plugin():
    def __init__(self, window):
        self.name = "render" # change if the plugin works for another application
        self.window = window

    def execute(self):
        """
        This function is called when opening a GUI.

        It should add buttons to the menu bar, connect such actions
        to fucntions, etc.
        """

        pass