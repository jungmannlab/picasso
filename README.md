# Picasso
A collection of tools for painting super-resolution images. The Picasso software is complemented by our Nature Protocols publication (https://www.nature.com/nprot/journal/v12/n6/abs/nprot.2017.024.html).

## Installation
Download and run the one-click installer file for Windows from [http://www.jungmannlab.org/](http://www.jungmannlab.org). Note that this is the Picasso version presented in the Nature Protocols publication. Feel free to reach out via picasso@jungmannlab.org to get the latest installer of the current commit. For the platform-independent usage of Picasso (e.g., with Linux and Mac Os X) please follow the advanced installation instructions.

### Advanced installation for Python programmers
As an alternative to the stand-alone program for end-users, Picasso can be installed as a Python package. This is the preferred option to use Picasso's internal routines in custom Python programs. For windows, one is still possible to use Picasso as an end-user by creating the respective shortcuts. This allows Picasso to be used on the same system by both programmers and end-users.

#### Requirements

##### Python 3.* (currently tested with 3.5)  
We highly recommend the [Anaconda or Miniconda](https://www.continuum.io/downloads) Python distribution which comes with a powerful package manager.

##### Python packages
The following packages are required:  
`h5py matplotlib numba numpy scipy pyqt=4 pyyaml scikit-learn colorama lmfit tqdm`  
When using Anaconda or Miniconda, most can be installed via `conda install <package>`. However, some packages need to be obtained from third-party conda channels. Visit [anaconda.org](anaconda.org) to search for them. Use `pip`  as a last resort to install packages from [PyPi](https://pypi.python.org/pypi). See instructions below as reference:

##### Creating an environment with conda
Sample instructions to create an environment with conda are as follows:
1. Open the console and create a new conda environment: `conda create --name picasso python=3.5`
2. Activate the environment: `source activate picasso`
3. Install the necessary packages with conda: `conda install h5py matplotlib numba numpy scipy pyqt=4 pyyaml scikit-learn colorama tqdm`
4. Additionally install packages with pip: `pip install lmfit`
5. Continue with Installation 

#### Installation
1. Open the console, `cd` to the directory where you want to install and run
`git clone https://github.com/jungmannlab/picasso`
Alternatively, [download](https://github.com/jungmannlab/picasso/archive/master.zip) the zip file and unzip it.
2. Change to the downloaded directory and run `python setup.py install`.

#### Optional for windows users
Run the PowerShell script "createShortcuts.ps1" in the gui directory.
This should be doable by right-clicking on the script and choosing "Run with PowerShell". Alternatively, run the command `powershell ./createShortcuts.ps1` in the command line. Use the generated shortcuts in the top level directory to start GUI components. Users can drag these shortcuts to their Desktop, Start Menu or Task Bar.

#### Using Picasso as a module
The individual modules of picasso can be started as follows:
1. Open the console, activate the enviroment: `source activate picasso`
2. Start the picasso modules via `python -m picasso ..`, e.g. `python -m picasso render` for the render module

#### Using GPU for Fitting
To enable GPU fitting, follow instructions on [Gpufit](https://github.com/gpufit/Gpufit) to install the Gpufit python library in your conda environment. Picasso Localize will automatically import the library if present and enables a checkbox for GPU fitting when selecting the LQ-Method.

## Contributions & Copyright
Contributors: Joerg Schnitzbauer, Maximilian Strauss  
Copyright (c) 2015-2018 Jungmann Lab, Max Planck Institute of Biochemistry

## Credits
- Design icon based on "Hexagon by Creative Stalls from the Noun Project"
- Simulate icon based on "Microchip by Futishia from the Nount Project"
- Localize icon based on "Mountains by MONTANA RUCOBO from the Noun Project"
- Filter icon based on "Funnel by José Campos from the Noun Project"
- Render icon based on "Paint Palette by Vectors Market from the Noun Project"
- Average icon based on "Layers by Creative Stall from the Noun Project"
