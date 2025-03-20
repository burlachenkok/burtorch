#pragma once

#if BURT_LINUX || BURT_MACOS
#include "burt/timers/include/linux/GetTimeOfDayTimer.h"
#elif BURT_WINDOWS
#include "burt/timers/include/windows/HiPrecOueryPerfomanceTimer.h"
#endif

namespace burt
{
#if BURT_LINUX || BURT_MACOS
    typedef posix::GetTimeOfDayTimer HighPrecisionTimer;
#elif BURT_WINDOWS
    typedef windows::HiPrecOueryPerfomanceTimer HighPrecisionTimer;
#endif
}
