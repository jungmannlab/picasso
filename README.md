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
| tifffile | no                   | [Download](http://www.lfd.uci.edu/~gohlke/pythonlibs/), then `pip install *.whl` |
| numba    | yes                  | pre-installed             |
| PyQt4    | yes                  | pre-installed             |

## Installation
1. `git clone https://gitlab.com/jungmannlab/picasso.git` or [download](https://gitlab.com/jungmannlab/picasso) the zip file (you need approved access).
2. Copy `picasso.pth` to your `site-packages` folder in your Python installation directory.
3. Add `picasso/scripts` to your PATH environment variable.
4. Windows only: (the equivalent of Linux or OSX procedures is not documented here.)
  - Add `.PY;.PYW` to your `PATHEXT` environment variable
  - Run these commands in an Administrator console:
    - `assoc .py=Python.File`
    - `assoc .pyw=Python.NoConFile`
    - `ftype Python.File=C:\Anaconda3\python.exe %1` (replace path if you don't use that Anaconda version)
    - `ftype Python.NoConFile=C:\Anaconda3\pythonw.exe %1` (same here)
