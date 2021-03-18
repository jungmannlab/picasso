# Packaging Picasso and Creating an Installer
This document describes the procedure to generate a Windows installer for Picasso end-users. The result is that Picasso (and Python) is installed in a single folder, the command line interface is exposed and start menu shortcuts are created. The first step is to package Picasso and its underlying Python distribution into a single folder.

## Requirements
- A Miniconda or Anaconda Python installation with a separate environment called `picasso` (See Picasso readme). In this environment should be installed:
  - Picasso requirements
  - Additionally install Pyinstaller (pip install pyinstaller)
- Inno Setup (free program, download it from the internet)

## Creating the Installer
1. Navigate to ./picasso/distribution
2. Make sure that in `create_installer.bat`, the Inno Setup executable points to the correct file installation path.
3. Run `create_installer.bat`

You will now find the installer executable in the folder `Output`, ready for distribution.

# Creating shortcuts for Linux
The script `create_linux_shortcuts.py` creates application menu entries (`*.desktop` files following the desktop entry specification by freedesktop.org) and an executable that can be called from the terminal without explicitly activating the picasso environment.

1. Clone/download picasso, set up the picasso environment using `python -m venv path/to/environment`, activate the environment, install the requirements and run `python setup.py install`.
2. With the picasso environment activated, run the script `create_linux_shortcuts.py`.

When `create_linux_shortcuts.py` is run as root, the picasso script is created as `/usr/bin/picasso` and the desktop files are created in `/usr/share/applications`.
For non-root users, the picasso script is created as `~/bin/picasso`, assuming that `~/bin` is on your `$PATH`, and the desktop files are created in `~/.local/share/applications`.
The picasso script can then be run without explicitly activating the picasso environment.
Note that this setup was only tested with environments created using venv; it may not work with conda environments.
