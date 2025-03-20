#include "linux/GetTimeOfDayTimer.h"

#if BURT_LINUX || BURT_MACOS
#include <sys/time.h>

namespace burt
{
    namespace posix
    {
        GetTimeOfDayTimer::GetTimeOfDayTimer()
        {
            gettimeofday(&lastTickTime, NULL);
        }

        double GetTimeOfDayTimer::measureTimeResolutionInmilliseconds()
        {
            GetTimeOfDayTimer tmp;
            return measureTimeResolutionHelper(tmp);
        }

        void GetTimeOfDayTimer::doSaveCurrentTickStateInPrev()
        {
            gettimeofday(&lastTickTime, NULL);
        }

        double GetTimeOfDayTimer::getDelatMsFromLastTickState()
        {
            timeval curTickTime = {};
            gettimeofday(&curTickTime, NULL);

            double elapsedTime = (curTickTime.tv_sec - lastTickTime.tv_sec) * 1000.0;   // sec to ms
            elapsedTime += (curTickTime.tv_usec - lastTickTime.tv_usec) / 1000.0;       // us to ms
            return elapsedTime;
        }
    }
}
#endif
