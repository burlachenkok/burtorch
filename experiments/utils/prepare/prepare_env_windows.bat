
echo Step-1: Setting CPU performance to maximum

powercfg -setactive SCHEME_MIN
powercfg /list

echo Step-1: Stop not-need services

sc stop wuauserv
sc stop bits
sc stop SysMain

::pskill.exe python.exe
:: powercfg.cpl -- check power mode
