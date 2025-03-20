#pragma once

#include "burt/timers/include/BaseTimer.h"

#if BURT_LINUX || BURT_MACOS

#include <sys/time.h>

namespace burt
{
    namespace posix
    {
        /** Timer with implementation based on Posix function gettimeofday()
        * + system-wide wall-clock time, not a process-specific time or thread specific
        * + resolution is microseconds
        */
        class GetTimeOfDayTimer final: public BaseTimer
        {
        public:
            GetTimeOfDayTimer();

            /** Time resolution for this timer
            * @return experimentally measured minimum time step
            */
            static double measureTimeResolutionInmilliseconds();

        protected:
            virtual void doSaveCurrentTickStateInPrev();
            virtual double getDelatMsFromLastTickState();
        private:
            timeval lastTickTime;
        };
    }
}

#endif
