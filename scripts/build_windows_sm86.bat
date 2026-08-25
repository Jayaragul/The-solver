@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

if not exist "%ROOT%\build\native" mkdir "%ROOT%\build\native"
pushd "%ROOT%"

cl /nologo /std:c11 /I c\include /c c\src\sankhya_model.c /Fo:build\native\sankhya_model.obj || exit /b 1
cl /nologo /std:c11 /I c\include /c c\src\sankhya_lu.c /Fo:build\native\sankhya_lu.obj || exit /b 1
cl /nologo /std:c11 /I c\include /c c\src\sankhya_mps.c /Fo:build\native\sankhya_mps.obj || exit /b 1
cl /nologo /std:c11 /I c\include /c c\src\sankhya_verify.c /Fo:build\native\sankhya_verify.obj || exit /b 1
cl /nologo /std:c11 /I c\include /c c\src\sk_lu.c /Fo:build\native\sk_lu.obj || exit /b 1
cl /nologo /std:c11 /I c\include /c c\src\sk_core.c /Fo:build\native\sk_core.obj || exit /b 1
cl /nologo /std:c11 /I c\include /c c\src\sk_solve.c /Fo:build\native\sk_solve.obj || exit /b 1
cl /nologo /std:c11 /I c\include apps\sankhya_solve.c build\native\sankhya_model.obj build\native\sankhya_lu.obj build\native\sankhya_mps.obj build\native\sankhya_verify.obj build\native\sk_lu.obj build\native\sk_core.obj build\native\sk_solve.obj /Fe:build\native\sankhya_solve.exe || exit /b 1
nvcc -arch=sm_86 -std=c++17 -I cuda\include -c cuda\src\sankhya_cuda.cu -o build\native\sankhya_cuda.obj || exit /b 1
nvcc -arch=sm_86 -std=c++17 -I cuda\include -c cuda\src\sankhya_cuda_lp.cu -o build\native\sankhya_cuda_lp.obj || exit /b 1
nvcc -arch=sm_86 -std=c++17 -I c\include -I cuda\include apps\sankhya_pdhg.cu build\native\sankhya_model.obj build\native\sankhya_lu.obj build\native\sankhya_mps.obj build\native\sankhya_verify.obj build\native\sankhya_cuda.obj build\native\sankhya_cuda_lp.obj -o build\native\sankhya_pdhg.exe || exit /b 1

popd
echo Built build\native\sankhya_pdhg.exe
