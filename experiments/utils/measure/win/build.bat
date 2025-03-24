call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64
cl.exe /std:c++latest /MT /Ot /GL /O2 measure.cpp
del measure.obj
