call DEL /F/Q/S build > NUL
call DEL /F/Q/S dist > NUL
call RMDIR /Q/S build
call RMDIR /Q/S dist

call cd %~dp0\..\..

call conda create -n picasso_installer python=3.10.19 -y
call conda activate picasso_installer

call pip install build
call python -m build

call cd release/one_click_windows_gui
call pip install "../../dist/picassosr-0.9.6-py3-none-any.whl"

call pip install pyinstaller==6.19.0
call pyinstaller ../pyinstaller/picasso.spec -y --clean
call pyinstaller ../pyinstaller/picassow.spec -y --clean
call conda deactivate
call conda remove -n picasso_installer --all -y

call robocopy ../../picasso dist/picasso/_internal/picasso /E
call robocopy ../../picasso dist/picassow/_internal/picasso /E

copy dist\picassow\picassow.exe dist\picasso\picassow.exe
copy dist\picassow\picassow.exe.manifest dist\picasso\picassow.exe.manifest

call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" picasso_innoinstaller.iss