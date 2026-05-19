call DEL /F/Q/S build > NUL
call DEL /F/Q/S dist > NUL
call RMDIR /Q/S build
call RMDIR /Q/S dist

call cd %~dp0\..\..

call conda create -n picasso_installer python=3.14.4 -y
call conda activate picasso_installer
call pip install build
call python -m build

for /f %%i in ('python -c "exec(open('picasso/version.py').read()); print(__version__)"') do set PICASSO_VERSION=%%i
call pip install "dist/picassosr-%PICASSO_VERSION%-py3-none-any.whl[installer]"
call cd release/one_click_windows_gui

call pyinstaller "../pyinstaller/picasso_pyinstaller.py" ^
    --onedir ^
    --collect-all picasso ^
    --collect-all PyImarisWriter ^
    --collect-all streamlit ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-submodules matplotlib.backends ^
    --copy-metadata streamlit ^
    --copy-metadata imageio ^
    --name picasso ^
    --icon "../logos/localize.ico" ^
    --noconfirm
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" ^
    --onedir ^
    --windowed ^
    --collect-all picasso ^
    --collect-all PyImarisWriter ^
    --collect-all streamlit ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-submodules matplotlib.backends ^
    --copy-metadata streamlit ^
    --copy-metadata imageio ^
    --name picassow ^
    --icon "../logos/localize.ico" ^
    --noconfirm

call DEL /F/Q picasso.spec
call DEL /F/Q picassow.spec

call conda deactivate
call conda env remove -n picasso_installer -y

copy dist\picassow\picassow.exe dist\picasso\picassow.exe
call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAPP_VERSION=%PICASSO_VERSION% picasso_innoinstaller.iss
