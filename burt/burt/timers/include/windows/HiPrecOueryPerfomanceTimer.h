#pragma once

#include "burt/timers/include/BaseTimer.h"

#ifdef BURT_WINDOWS

#include <windows.h>

namespace burt
{
    namespace windows
    {
        /** Timer with implementation based on WinApi function OueryPerfomanceCounter()
        * + Very good for very small time measurements in small blocks of code
        * - Very bad for big time measurements.
        * - Absolutely not correct if process will be displaced by others during scheduling
        */
        class HiPrecOueryPerfomanceTimer final: public BaseTimer
        {
        public:
            HiPrecOueryPerfomanceTimer();
            static double measureTimeResolutionInmilliseconds();

        protected:
            virtual void doSaveCurrentTickStateInPrev();
            virtual double getDelatMsFromLastTickState();

        private:
            LARGE_INTEGER perfCounterFrequency;
            LARGE_INTEGER perfCounterValue;
        };
    }
}

#endif
