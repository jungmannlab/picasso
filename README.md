# Picasso
A collection of tools for painting super-resolution images

## Requirements
### Python 3.* (tested with 3.5)  
I suggest installing it with [Anaconda](https://www.continuum.io/downloads) which comes bundled with many useful third-party packages and a package manager which removes the pain of building some packages by yourself.

### Python packages
Various Python packages need to be installed. Find out which by looking at either the source code or error messages when running Picasso programs.

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
5. Run this command to register the picasso package in the Python installation.
    - `ECHO <picasso directory> >> <python installation directory>\Lib\site-packages\picasso.pth`
6. Run these commands in an Administrator console to tell Windows that it should run .py and .pyw files with the Python interpreter:
    - `assoc .py=Python.File`
    - `assoc .pyw=Python.NoConFile`
    - `ftype Python.File=<python installation directory>\python.exe %1`
    - `ftype Python.NoConFile=<python installation directory>\pythonw.exe %1`  
7. Make shortcuts of `gui\*.pyw` files. Open shortcut properties. Prepend shortcut command with `pythonw `. Set the icon to the respective file in the `gui` folder. Move the shortcut to top level. This shortcut can now be double-clicked, pinned to task bar or copied to Desktop.


## Credits
- Localize icon based on "Mountains by MONTANA RUCOBO from the Noun Project"
- ToRaw icon based on "Lion by Sathish Selladurai from the Noun Project"
- Filter icon based on "Funnel by José Campos from the Noun Project"
- Render icon based on "Paint Palette by Vectors Market from the Noun Project"
