# Picasso
A collection of tools for painting super-resolution images

## Requirements
### Python 3.* (tested with 3.5)
I suggest installing it with [Anaconda / Miniconda](https://www.continuum.io/downloads) which comes bundled with many useful third-party packages and a package manager which removes the pain of building some packages by yourself.

### Python packages
A few third-party Python packages are required. It is recommended to install them via `conda install <package>`.  
You might need to obtain packages from third-party conda channels. Visit [anaconda.org](anaconda.org) to search for them.  
The following packages are required:  
`h5py matplotlib numba numpy scipy pyqt pyyaml scikit-learn lmfit tqdm`

## Installation

1. Open the console, `cd` to the directory where you want to install and run
`git clone https://gitlab.com/jungmannlab/picasso.git`
Alternatively, [download](https://gitlab.com/jungmannlab/picasso) the zip file and unzip it.
For both options you need approved access on Gitlab.
2. `python setup.py install`
3. Run the PowerShell script "createShortcuts.ps1" in the gui directory.
This should be doable by right-clicking on the script and choosing "Run with PowerShell". Alternatively, run the command `powershell ./createShortcuts.ps1` in the command line.
Use the generated shortcuts in the top level directory to start GUI components.
You can drag these shortcuts to the Desktop, Start Menu or Task Bar.

## Contributions & Copyright
Contributors: Joerg Schnitzbauer, Maximilian Strauss
Copyright (c) 2015 Jungmann Lab, Max Planck Institute of Biochemistry

## Credits
- Localize icon based on "Mountains by MONTANA RUCOBO from the Noun Project"
- ToRaw icon based on "Lion by Sathish Selladurai from the Noun Project"
- Filter icon based on "Funnel by José Campos from the Noun Project"
- Render icon based on "Paint Palette by Vectors Market from the Noun Project"
