call DEL /F/Q/S build > NUL
call DEL /F/Q/S dist > NUL
call RMDIR /Q/S build
call RMDIR /Q/S dist

call cd %~dp0\..\..

call conda create -n picasso_installer python=3.10.19 -y
call conda activate picasso_installer
call pip install build
call python -m build

call pip install "dist/picassosr-0.9.7-py3-none-any.whl"
call cd release/one_click_windows_gui

call pip install pyinstaller==6.19.0
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" --onedir --collect-all picasso --collect-all PyImarisWriter --name picasso --icon "../logos/localize.ico" --noconfirm
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" --onedir --windowed --collect-all picasso --collect-all PyImarisWriter --name picassow --icon "../logos/localize.ico" --noconfirm

call conda deactivate
call conda remove -n picasso_installer --all -y

copy dist\picassow\picassow.exe dist\picasso\picassow.exe
call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" picasso_innoinstaller.iss