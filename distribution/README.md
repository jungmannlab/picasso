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