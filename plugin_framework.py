"""
Framework for creating a Picasso plugin. Any plugin should be moved to
picasso/picasso/gui/plugins/
Author:
Date:
"""

# Space to import packages
import numpy as np

# Do not change the part below unless stated otherwise
class Plugin():
	def __init___(self, window):
		self.name = "render" # change if the plugin works for another application
		self.window = window

	def execute():
# Specify the functionalities of your plugin in the space below