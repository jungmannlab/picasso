REM CUDA 13 edition: the CUDA 12 edition (create_installer_windows_cuda12.bat)
REM with the CUDA 13 runtime instead of CUDA 12, plus VectorFit's GPU build
REM ('vectorfit-gpu' fit method). Runs on NVIDIA drivers R580+ and GPUs of
REM compute capability >= 7.5 (Turing or newer) only; older ones need the
REM CUDA 12 edition. Keep the two scripts in step: they differ only in the
REM CUDA-13-specific parts below. The edition's name (-CUDA13) is how the
REM in-app updater (picasso/updater.py, find_installer) tells the two apart
REM in a release, so keep "CUDA13" in the file names.
REM Building needs no CUDA 13 driver (the CUDA runtime comes from pip
REM wheels), so it works on any Windows PC. The smoke test does not use the
REM GPU, though: check the GPU (numba.cuda and VectorFit GPU) fitting on a
REM PC with an R580+ driver before releasing.
call DEL /F/Q/S build_cuda13 > NUL
call DEL /F/Q/S dist_cuda13 > NUL
call RMDIR /Q/S build_cuda13
call RMDIR /Q/S dist_cuda13

call cd %~dp0\..\..

call conda create -n picasso_installer_cuda13 python=3.14.4 -y
call conda activate picasso_installer_cuda13

for /f %%i in ('python -c "exec(open('picasso/version.py').read()); print(__version__)"') do set PICASSO_VERSION=%%i
REM Picasso is built from a temporary copy of the repository
REM (..\install_picasso.py): a build in the repository itself writes build\,
REM dist\ and picassosr.egg-info there, which create_installer_windows.bat, if
REM it runs at the same time, would overwrite.
REM installer_cuda13 adds numba-cuda[cu13], which pulls the nvidia-* CUDA 13
REM runtime wheels needed to run numba.cuda (GPU-accelerated) code, and
REM vectorfit-py-cu12, VectorFit's GPU build (built with CUDA 13 despite its
REM name).
call python release\install_picasso.py installer_cuda13
if errorlevel 1 exit /b 1
call cd release/one_click_windows_gui

REM CUDA build differs from the CPU build by the extra flags below:
REM   --collect-all numba_cuda : the CUDA target package for numba
REM   --collect-all nvidia     : the bundled CUDA runtime libraries; the CUDA
REM                              13 wheels put all DLLs in nvidia\cu13\bin\x86_64
REM                              (no Python package), which this still collects
REM                              at the same path, where cuda-pathfinder looks
REM   --collect-all cuda       : NVIDIA's official CUDA binding (cuda.bindings
REM                              and cuda.core). numba-cuda >= 0.24 uses this
REM                              binding to talk to the driver; without it
REM                              numba.cuda.is_available() fails and the app
REM                              silently reports "no GPU".
REM   --copy-metadata numba-cuda / cuda-bindings / cuda-core : these packages
REM                              are discovered via importlib metadata, so the
REM                              dist-info must ship in the frozen app.
REM   --additional-hooks-dir ../pyinstaller/hooks : cuda.core implements its CUDA
REM                              runtime as compiled Cython extensions under a
REM                              cu12/cu13 subpackage it merges in at import time
REM                              via __path__; --collect-all drops those .pyd, so
REM                              hook-cuda.core.py copies them to their real path.
REM                              hook-cuda.bindings.py / hook-cuda.core.py also
REM                              bundle the cuda_bindings.libs / cuda_core.libs
REM                              DLLs (delvewheel-vendored msvcp140); without them
REM                              the app fails with "DLL load failed while
REM                              importing runtime".
REM   --hidden-import vectorfit_py : VectorFit (GPU build, unlike the other
REM                              installers) is imported dynamically;
REM                              hook-vectorfit_py.py adds its extension,
REM                              metadata and vectorfit_py_cu12.libs DLLs.
REM Note: numba-cuda redirects "numba.cuda" to its own target via a
REM site-packages .pth file that PyInstaller does not run, so the frozen app
REM would keep Numba's built-in CUDA stub and GPU-accelerated (numba.cuda) code
REM would fall back to CPU or vanish. picasso/_numba_cuda_compat.py reinstalls
REM that redirect at import time (shipped via --collect-all picasso), so no extra
REM PyInstaller flag is needed here.
REM CUDA 13 artifacts go to _cuda13-suffixed folders (--distpath/--workpath/
REM --specpath) so they never clash with the other builds. The exe names
REM (--name) stay "picasso"/"picassow" so picasso_innoinstaller.iss is unchanged.
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" ^
    --onedir ^
    --distpath dist_cuda13 ^
    --workpath build_cuda13 ^
    --specpath build_cuda13 ^
    --collect-all picasso ^
    --collect-all PyImarisWriter ^
    --collect-all streamlit ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-all numba_cuda ^
    --collect-all nvidia ^
    --collect-all cuda ^
    --hidden-import vectorfit_py ^
    --additional-hooks-dir "../pyinstaller/hooks" ^
    --copy-metadata numba-cuda ^
    --copy-metadata cuda-bindings ^
    --copy-metadata cuda-core ^
    --collect-submodules matplotlib.backends ^
    --copy-metadata streamlit ^
    --copy-metadata imageio ^
    --name picasso ^
    --icon "../../logos/picasso.ico" ^
    --noconfirm
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" ^
    --onedir ^
    --windowed ^
    --distpath dist_cuda13 ^
    --workpath build_cuda13 ^
    --specpath build_cuda13 ^
    --collect-all picasso ^
    --collect-all PyImarisWriter ^
    --collect-all streamlit ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-all numba_cuda ^
    --collect-all nvidia ^
    --collect-all cuda ^
    --hidden-import vectorfit_py ^
    --additional-hooks-dir "../pyinstaller/hooks" ^
    --copy-metadata numba-cuda ^
    --copy-metadata cuda-bindings ^
    --copy-metadata cuda-core ^
    --collect-submodules matplotlib.backends ^
    --copy-metadata streamlit ^
    --copy-metadata imageio ^
    --name picassow ^
    --icon "../../logos/picasso.ico" ^
    --noconfirm

REM The installer redistributes Python, PyQt6/Qt, the CUDA runtime and every
REM other bundled library, whose licenses require shipping their notices with
REM it. Collect them while the build environment is still active so the list
REM matches the frozen packages; the file lands in the installation folder.
call python ..\collect_third_party_licenses.py dist_cuda13\picasso\THIRD-PARTY-LICENSES.txt
if errorlevel 1 exit /b 1

REM Test the frozen app before packaging it (..\smoke_test.py): a short
REM analysis on the command line and every GUI opened offscreen, also from
REM the windowed build. A failure stops here, so a broken build never becomes
REM an installer. Set PICASSO_SKIP_SMOKE_TEST=1 to skip the test.
if not defined PICASSO_SKIP_SMOKE_TEST (
    call python ..\smoke_test.py --frozen dist_cuda13\picasso\picasso.exe
    if errorlevel 1 exit /b 1
    call python ..\smoke_test.py --frozen dist_cuda13\picassow\picassow.exe --gui-only
    if errorlevel 1 exit /b 1
)

REM Spec files were written to build_cuda13 (via --specpath); they are removed with
REM the build_cuda13 tree below, so no separate *.spec deletion is needed here.

call conda deactivate
call conda env remove -n picasso_installer_cuda13 -y

copy dist_cuda13\picassow\picassow.exe dist_cuda13\picasso\picassow.exe
call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAPP_VERSION=%PICASSO_VERSION% /DVARIANT=-CUDA13 /DDISTDIR=dist_cuda13 picasso_innoinstaller.iss
if errorlevel 1 exit /b 1

REM Zip the installer (Inno Setup writes it to Output\) with readme.txt, for
REM the GitHub release
set INSTALLER=Output\Picasso-Windows-64bit-CUDA13-%PICASSO_VERSION%
powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; Compress-Archive -Path '%INSTALLER%.exe', 'readme.txt' -DestinationPath '%INSTALLER%.zip' -Force"
if errorlevel 1 exit /b 1
echo Created %INSTALLER%.zip
