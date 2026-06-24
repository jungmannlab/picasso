"""Template for creating a Picasso plugin. Save your plugin as a .py file in
~/.picasso/plugins/ (use Plugins -> Open plugins folder... in any Picasso app
to find it).

For more details, see https://picassosr.readthedocs.io/en/latest/plugins.html

Author:
Date:
"""

# Space to import packages
import numpy as np


# class that defines modifications to the GUI and actions
class Plugin:
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
        """This function is called when clicking the menu entry."""
        print(f"Hello world, pi is: {np.pi}")
