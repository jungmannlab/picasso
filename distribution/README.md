# Packaging Picasso and Creating an Installer
This document describes the procedure to generate a Windows installer for Picasso end-users. The result is that Picasso (and Python) is installed in a single folder, the command line interface is exposed and start menu shortcuts are created. The first step is to package Picasso and its underlying Python distribution into a single folder.

## Requirements
- A Miniconda or Anaconda Python installation with a separate environment called `picasso` (`conda create -n picasso python=3`). In this environment should be installed:
  - Picasso requirements
  - Pyinstaller (I had best experiences by installing the newest version from Github with pip)
- Inno Setup (free program, download it from the internet)

## Packaging
1. In `create_installer.bat`, make sure that the Inno Setup executable points to the correct file.
2. Run `create_installer.bat`

You will now find the installer executable in the folder `Output`, ready for distribution.
