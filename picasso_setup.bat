conda config --add channels conda-forge
conda install -y h5py lmfit matplotlib numba numpy pyqt pyyaml scikit-learn tifffile tqdm
git clone https://gitlab.com/jungmannlab/picasso.git C:\picasso
SETX /M PATH "%PATH%;C:\picasso\scripts"
ECHO C:\picasso >> C:\Miniconda3\Lib\site-packages\picasso.pth
powershell -ExecutionPolicy Bypass -File C:\picasso\picasso\gui\createShortcuts.ps1