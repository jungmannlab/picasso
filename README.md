# Picasso
A collection of tools for painting super-resolution images

## Installation
Simply download and run the installer:  
[Picasso-Windows-64bit.exe](# TODO enter link here) (Link needs to be updated)

### Advanced installation for Python programmers
As an alternative to the stand-alone program for end-users, Picasso can be installed as a Python package. This is the preferred option to use Picasso's internal routines in custom Python programs. At the same time, it is still possible to use Picasso as an end-user by creating the respective Windows shortcuts. This allows Picasso to be used on the same system by both programmers and end-users.

#### Requirements

##### Python 3.* (currently tested with 3.5)  
We highly recommend the [Anaconda or Miniconda](https://www.continuum.io/downloads) Python distribution which comes with a powerful package manager.

##### Python packages
The following packages are required:  
`h5py matplotlib numba numpy scipy pyqt pyyaml scikit-learn colorama lmfit tqdm`  
When using Anaconda or Miniconda, most can be installed via `conda install <package>`. However, some packages need to be obtained from third-party conda channels. Visit [anaconda.org](anaconda.org) to search for them. Use `pip`  as a last resort to install packages from [PyPi](https://pypi.python.org/pypi).

#### Installation

1. Open the console, `cd` to the directory where you want to install and run
`git clone https://gitlab.com/jungmannlab/picasso.git`
Alternatively, [download](https://gitlab.com/jungmannlab/picasso/repository/archive.zip?ref=master) the zip file and unzip it.
2. Change to the downloaded directory and run `python setup.py install`.
3. Run the PowerShell script "createShortcuts.ps1" in the gui directory.
This should be doable by right-clicking on the script and choosing "Run with PowerShell". Alternatively, run the command `powershell ./createShortcuts.ps1` in the command line. Use the generated shortcuts in the top level directory to start GUI components. Users can drag these shortcuts to their Desktop, Start Menu or Task Bar.

## Contributions & Copyright
Contributors: Joerg Schnitzbauer, Maximilian Strauss  
Copyright (c) 2015-2016 Jungmann Lab, Max Planck Institute of Biochemistry

## Credits
- Design icon based on "Hexagon by Creative Stalls from the Noun Project"
- Simulate icon based on "Microchip by Futishia from the Nount Project"
- Localize icon based on "Mountains by MONTANA RUCOBO from the Noun Project"
- ToRaw icon based on "Lion by Sathish Selladurai from the Noun Project"
- Filter icon based on "Funnel by José Campos from the Noun Project"
- Render icon based on "Paint Palette by Vectors Market from the Noun Project"
- Average icon based on "Layers by Creative Stall from the Noun Project"
