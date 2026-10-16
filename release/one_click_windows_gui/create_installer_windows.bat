call DEL /F/Q/S build > NUL
call DEL /F/Q/S dist > NUL
call RMDIR /Q/S build
call RMDIR /Q/S dist

call cd %~dp0\..\..

call conda create -n picasso_installer python=3.14.4 -y
call conda activate picasso_installer

for /f %%i in ('python -c "exec(open('picasso/version.py').read()); print(__version__)"') do set PICASSO_VERSION=%%i
REM Picasso is built from a temporary copy of the repository
REM (..\install_picasso.py): a build in the repository itself writes build\,
REM dist\ and picassosr.egg-info there, which create_installer_windows_cuda.bat,
REM if it runs at the same time, would overwrite.
call python release\install_picasso.py installer
if errorlevel 1 exit /b 1
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
    --icon "../logos/picasso.ico" ^
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
    --icon "../logos/picasso.ico" ^
    --noconfirm

REM The installer redistributes Python, PyQt6/Qt and every other bundled
REM library, whose licenses require shipping their notices with it. Collect
REM them while the build environment is still active so the list matches the
REM frozen packages; the file lands in the installation folder (C:\Picasso).
call python ..\collect_third_party_licenses.py dist\picasso\THIRD-PARTY-LICENSES.txt
if errorlevel 1 exit /b 1

REM Test the frozen app before packaging it (..\smoke_test.py): a short
REM analysis on the command line and every GUI opened offscreen, also from
REM the windowed build. A failure stops here, so a broken build never becomes
REM an installer. Set PICASSO_SKIP_SMOKE_TEST=1 to skip the test.
if not defined PICASSO_SKIP_SMOKE_TEST (
    call python ..\smoke_test.py --frozen dist\picasso\picasso.exe
    if errorlevel 1 exit /b 1
    call python ..\smoke_test.py --frozen dist\picassow\picassow.exe --gui-only
    if errorlevel 1 exit /b 1
)

call DEL /F/Q picasso.spec
call DEL /F/Q picassow.spec

call conda deactivate
call conda env remove -n picasso_installer -y

copy dist\picassow\picassow.exe dist\picasso\picassow.exe
call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAPP_VERSION=%PICASSO_VERSION% picasso_innoinstaller.iss
if errorlevel 1 exit /b 1

REM Zip the installer (Inno Setup writes it to Output\) with readme.txt, for
REM the GitHub release
set INSTALLER=Output\Picasso-Windows-64bit-%PICASSO_VERSION%
powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; Compress-Archive -Path '%INSTALLER%.exe', 'readme.txt' -DestinationPath '%INSTALLER%.zip' -Force"
if errorlevel 1 exit /b 1
echo Created %INSTALLER%.zip
