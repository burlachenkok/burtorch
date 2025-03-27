call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64

set LIBTORCH_PATH=C:\Users\burla\Downloads\old\example_vs_2022\libtorch-win-shared-with-deps-2.5.1+cpu

cl.exe ^
/I"%LIBTORCH_PATH%\libtorch\include" ^
/I"%LIBTORCH_PATH%\libtorch\include\torch\csrc\api\include" ^
/std:c++latest /MD /Ot /GL /O2 /std:c++17 ^
main.cpp ^
/link ^
/SUBSYSTEM:CONSOLE ^
/MACHINE:X64 ^
/LIBPATH:"%LIBTORCH_PATH%\libtorch\lib" ^
"asmjit.lib" "c10.lib" "cpuinfo.lib" "dnnl.lib" "fbgemm.lib" "fmt.lib" "kineto.lib" "libprotobuf.lib" "libprotoc.lib" "pthreadpool.lib" "torch.lib" "torch_cpu.lib" "XNNPACK.lib" "kernel32.lib" "user32.lib" "gdi32.lib" "winspool.lib" "comdlg32.lib" "advapi32.lib" "shell32.lib" "ole32.lib" "oleaut32.lib" "uuid.lib" "odbc32.lib" "odbccp32.lib"

cp "%LIBTORCH_PATH%/libtorch/lib/"* ./
