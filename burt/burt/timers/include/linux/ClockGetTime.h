#pragma once

#include "burt/timers/include/BaseTimer.h"

#if BURT_LINUX || BURT_MACOS

#include <sys/time.h>

namespace burt
{
    namespace posix
    {
        /** Timer with implementation based on Posix function clock_gettime(CLOCK_MONOTONIC)
        * + Guarantees never run backwards
        * + Call is pretty fast. In practice faster by x100 compare to system call in Linux.
        * + Not per process, not per thread. It's per whole system.
        * + resolution is nanoseconds
        * @note https://pubs.opengroup.org/onlinepubs/9699919799/functions/clock_gettime.html
        */
        class ClockGetTime final: public BaseTimer
        {
        public:
            ClockGetTime();

            /** Time resolution for this timer
            * @return experimentally measured minimum time step
            */
            static double measureTimeResolutionInmilliseconds();

        protected:
            virtual void doSaveCurrentTickStateInPrev();
            virtual double getDelatMsFromLastTickState();
        private:
            timespec lastTickTime;
        };
    }
}

#endif
