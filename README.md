# Picasso
A collection of tools for painting super-resolution images

## Requirements
### Python 3.* (tested with 3.4)  
I suggest installing it with [Anaconda](https://www.continuum.io/downloads) which comes bundled with many useful third-party packages and a package manager which removes the pain of building some packages by yourself.

### Python third-party packages
| Package  | Included in Anaconda | Installation instructions |
| -------- | -------------------- | ------------------------- |
| numpy    | yes                  | pre-installed             |
| pyyaml   | yes                  | pre-installed             |
| numba    | yes                  | pre-installed             |
| PyQt4    | yes                  | pre-installed             |
| tifffile | no                   | [Download](http://www.lfd.uci.edu/~gohlke/pythonlibs/), then `pip install *.whl` |

## Installation
The described procedure is intended for Windows. The equivalent steps for Linux or OSX are not documented.
Replace any <...> notations according to your situation.
1. Open the console, `cd` to the directory where you want to install and run  
`git clone https://gitlab.com/jungmannlab/picasso.git`  
Alternatively, [download](https://gitlab.com/jungmannlab/picasso) the zip file and unzip it.  
For both options you need approved access on Gitlab.
3. Add `<picasso directory>\scripts` to your PATH environment variable.  
This will make the scripts in this folder accessible in the
console, independent of the current directory.
4. Add `.PY;.PYW` to your `PATHEXT` environment variable.  
The result is that you don't have to specify the .py or .pyw ending when you run a Python in the command line.
5. Run these commands in an Administrator console to tell Windows that it should run .py and .pyw files with the Python interpreter:
    - `assoc .py=Python.File`
    - `assoc .pyw=Python.NoConFile`
    - `ftype Python.File=<python installation directory>\python.exe %1`
    - `ftype Python.NoConFile=<python installation directory>\pythonw.exe %1`
6. Run this command to register the picasso package in the Python installation.
    - `ECHO <picasso directory> >> <python installation directory>\Lib\site-packages\picasso.pth`
