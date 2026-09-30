call DEL /F/Q/S build_cuda > NUL
call DEL /F/Q/S dist_cuda > NUL
call RMDIR /Q/S build_cuda
call RMDIR /Q/S dist_cuda

call cd %~dp0\..\..

call conda create -n picasso_installer_cuda python=3.14.4 -y
call conda activate picasso_installer_cuda
call pip install build

for /f %%i in ('python -c "exec(open('picasso/version.py').read()); print(__version__)"') do set PICASSO_VERSION=%%i
REM The CPU and CUDA installers share the repo-level dist/ folder. If both are
REM created at the same time, the second "python -m build" run fails because the
REM sdist/wheel already exist, so only build when they are missing.
if exist "dist\picassosr-%PICASSO_VERSION%-py3-none-any.whl" (
    echo Reusing existing dist/picassosr-%PICASSO_VERSION%-py3-none-any.whl
) else (
    call python -m build
)
REM installer_cuda adds numba-cuda[cu12], which pulls the nvidia-*-cu12 CUDA
REM runtime wheels needed to run numba.cuda (GPU-accelerated) code.
call pip install "dist/picassosr-%PICASSO_VERSION%-py3-none-any.whl[installer_cuda]"
call cd release/one_click_windows_gui

REM CUDA build differs from the CPU build by the extra flags below:
REM   --collect-all numba_cuda : the CUDA target package for numba
REM   --collect-all nvidia     : the bundled CUDA runtime libraries (cu12)
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
REM Note: numba-cuda redirects "numba.cuda" to its own target via a
REM site-packages .pth file that PyInstaller does not run, so the frozen app
REM would keep Numba's built-in CUDA stub and GPU-accelerated (numba.cuda) code
REM would fall back to CPU or vanish. picasso/_numba_cuda_compat.py reinstalls
REM that redirect at import time (shipped via --collect-all picasso), so no extra
REM PyInstaller flag is needed here.
REM CUDA artifacts go to _gpu-suffixed folders (--distpath/--workpath/--specpath)
REM so they never clash with the CPU build's build/dist/*.spec. The exe names
REM (--name) stay "picasso"/"picassow" so picasso_innoinstaller.iss is unchanged.
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" ^
    --onedir ^
    --distpath dist_cuda ^
    --workpath build_cuda ^
    --specpath build_cuda ^
    --collect-all picasso ^
    --collect-all PyImarisWriter ^
    --collect-all streamlit ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-all numba_cuda ^
    --collect-all nvidia ^
    --collect-all cuda ^
    --additional-hooks-dir "../pyinstaller/hooks" ^
    --copy-metadata numba-cuda ^
    --copy-metadata cuda-bindings ^
    --copy-metadata cuda-core ^
    --collect-submodules matplotlib.backends ^
    --copy-metadata streamlit ^
    --copy-metadata imageio ^
    --name picasso ^
    --icon "../../logos/localize.ico" ^
    --noconfirm
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" ^
    --onedir ^
    --windowed ^
    --distpath dist_cuda ^
    --workpath build_cuda ^
    --specpath build_cuda ^
    --collect-all picasso ^
    --collect-all PyImarisWriter ^
    --collect-all streamlit ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-all numba_cuda ^
    --collect-all nvidia ^
    --collect-all cuda ^
    --additional-hooks-dir "../pyinstaller/hooks" ^
    --copy-metadata numba-cuda ^
    --copy-metadata cuda-bindings ^
    --copy-metadata cuda-core ^
    --collect-submodules matplotlib.backends ^
    --copy-metadata streamlit ^
    --copy-metadata imageio ^
    --name picassow ^
    --icon "../../logos/localize.ico" ^
    --noconfirm

REM Spec files were written to build_cuda (via --specpath); they are removed with
REM the build_cuda tree below, so no separate *.spec deletion is needed here.

call conda deactivate
call conda env remove -n picasso_installer_cuda -y

copy dist_cuda\picassow\picassow.exe dist_cuda\picasso\picassow.exe
call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAPP_VERSION=%PICASSO_VERSION% /DVARIANT=-CUDA /DDISTDIR=dist_cuda picasso_innoinstaller.iss
