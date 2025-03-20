#include "linux/ClockGetTime.h"

#if BURT_LINUX || BURT_MACOS

#include <sys/time.h>

namespace burt
{
    namespace posix
    {
        ClockGetTime::ClockGetTime()
        {
            clock_gettime(CLOCK_MONOTONIC, &lastTickTime);
        }

        double ClockGetTime::measureTimeResolutionInmilliseconds()
        {
            ClockGetTime tmp;
            return measureTimeResolutionHelper(tmp);
        }

        void ClockGetTime::doSaveCurrentTickStateInPrev()
        {
            clock_gettime(CLOCK_MONOTONIC, &lastTickTime);
        }

        double ClockGetTime::getDelatMsFromLastTickState()
        {
            timespec curTickTime = {};
            clock_gettime(CLOCK_MONOTONIC, &curTickTime);

            double elapsedTime = (curTickTime.tv_sec - lastTickTime.tv_sec) * 1000.0;   // sec to ms
            elapsedTime += (curTickTime.tv_nsec - lastTickTime.tv_nsec) * (1.0e-6);     // nanoseconds to milliseconds
            return elapsedTime;
        }
    }
}
#endif
