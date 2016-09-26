# Packaging Picasso and Creating an Installer
This document describes the procedure to generate a Windows installer for Picasso end-users. The result is that Picasso (and Python) is installed in a single folder, the command line interface is exposed and start menu shortcuts are created. The first step is to package Picasso and its underlying Python distribution into a single folder.

## Requirements
- A Python installation into which the picasso package and its requirements have been installed. Ideally, this installation contains no more than the required packages for Picasso. conda environments might come in handy.
- Pyinstaller (I had best experiences by installing the newest version from Github with pip)
- Inno Setup (free program, download it from the internet)

## Packaging
We will build two distributions of picasso: one with console window and one without (for the command line interface and GUI programs, respectively). The two executables will be called picasso.exe and picassow.exe. Since picassow.exe depends on the exact same files than picasso.exe, we can simply copy picassow.exe from its distribution folder into the folder of picasso.exe. This way we will have both executables in the same directory, bundled with one Python distribution.

1. If you haven't yet, create a separate conda environment with `conda create -n picasso python=3`. Activate the environment.

2. Install the required Python packages for Picasso as well as Picasso itself into the environment.

3. Install PyInstaller into the environment.

4. Generate the executable with console window:  
`pyinstaller --hidden-import=h5py.defs --hidden-import=h5py.utils  --hidden-import=h5py.h5ac --hidden-import=h5py._proxy --hidden-import=sklearn.neighbors.typedefs -n picasso picasso-script.py`

5. And one without console window:  
`pyinstaller --hidden-import=h5py.defs --hidden-import=h5py.utils  --hidden-import=h5py.h5ac --hidden-import=h5py._proxy --hidden-import=sklearn.neighbors.typedefs --noconsole -n picassow picasso-script.py`

6. Copy `picassow.exe` and `picassow.exe.manifest` from `dist\picassow` to `dist\picasso`.

The folder `dist\picasso` now contains all files to run Picasso standalone (through picasso.exe or picassow.exe).

## Creating the Installer
In this step, we create a Windows installer that copies the picasso distribution folder to an installation destination, sets up the PATH environment for seamless command line usage of picasso.exe, and creates shortcuts for windowed Picasso elements via picassow.exe

1. Open Inno Setup.
2. Load the Inno Setup Script `create_installer.iss`.
3. Compile the script.

You will now find the installer executable in the folder `Output`, ready for distribution.
