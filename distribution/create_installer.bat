call activate picasso
cd %~dp0\..
python setup.py install
cd %~dp0
pyinstaller -y --hidden-import=h5py.defs --hidden-import=h5py.utils  --hidden-import=h5py.h5ac --hidden-import=h5py._proxy --hidden-import=sklearn.neighbors.typedefs -n picasso picasso-script.py
pyinstaller -y --hidden-import=h5py.defs --hidden-import=h5py.utils  --hidden-import=h5py.h5ac --hidden-import=h5py._proxy --hidden-import=sklearn.neighbors.typedefs --noconsole -n picassow picasso-script.py
copy dist\picassow\picassow.exe dist\picasso\picassow.exe
copy dist\picassow\picassow.exe.manifest dist\picasso\picassow.exe.manifest
"C:\Program Files (x86)\Inno Setup 5\ISCC.exe" picasso.iss
call deactivate
