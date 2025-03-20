#include "windows/HiPrecOueryPerfomanceTimer.h"

#ifdef BURT_WINDOWS

namespace burt
{
    namespace windows
    {
        HiPrecOueryPerfomanceTimer::HiPrecOueryPerfomanceTimer()
        {
            // The frequency of the performance counter is fixed at system boot and is consistent across all processors.
            // (https://msdn.microsoft.com/en-us/library/windows/desktop/ms644905(v=vs.85).aspx)
            QueryPerformanceFrequency(&perfCounterFrequency);
            doSaveCurrentTickStateInPrev();
        }

        double HiPrecOueryPerfomanceTimer::measureTimeResolutionInmilliseconds()
        {
            HiPrecOueryPerfomanceTimer tmp;
            return measureTimeResolutionHelper(tmp);
        }

        void HiPrecOueryPerfomanceTimer::doSaveCurrentTickStateInPrev()
        {
            QueryPerformanceCounter(&perfCounterValue);
        }

        double HiPrecOueryPerfomanceTimer::getDelatMsFromLastTickState()
        {
            // https://msdn.microsoft.com/ru-ru/library/windows/desktop/dn553408(v=vs.85).aspx
            LARGE_INTEGER perfCounterValueCur;
            QueryPerformanceCounter(&perfCounterValueCur);

            LARGE_INTEGER elapsedTime;
            elapsedTime.QuadPart = perfCounterValueCur.QuadPart - perfCounterValue.QuadPart;

            return double(elapsedTime.QuadPart * 1000/*000*/) / double(perfCounterFrequency.QuadPart);
        }
    }
}

#endif
