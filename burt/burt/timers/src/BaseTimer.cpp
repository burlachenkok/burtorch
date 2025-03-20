#include "BaseTimer.h"

#include <sstream>
#include <float.h>

namespace burt
{
    BaseTimer::BaseTimer()
    : processedTimeMs(0.0)
    , paused(0)
    , stampIndex(0)
    {}

    std::string BaseTimer::localTime()
    {
        time_t t = time(NULL);

#if BURT_WINDOWS
        tm tg_data = {};
        localtime_s(&tg_data, &t);
        tm* tg = &tg_data;
#else
        tm* tg = localtime(&t);
#endif
        std::stringstream s;
        s << tg->tm_hour << ':' << tg->tm_min << ':' << tg->tm_sec << " (time from the process start: " << clock() / double(CLOCKS_PER_SEC) << ")";
        return s.str();
    }

    double BaseTimer::getTimeMs()
    {
        const double r = getTimeMsInternal(false);
        return r;
    }

    double BaseTimer::getTimeSec()
    {
        const double r = getTimeMs() / 1000.0;
        return r;
    }

    double BaseTimer::getTimeMsInternal(bool forceUpProcessedTime)
    {
        if (isPaused())
        {
            doSaveCurrentTickStateInPrev();
            return processedTimeMs;
        }

        const double delta = getDelatMsFromLastTickState();

        if (delta < 1.0 && !forceUpProcessedTime)
        {
            // this schema uses assumtion that fraction of FP64 (52) is still senstive to add "1.0" or so
            //  => it is safe for 10^15 seconds
            return processedTimeMs + delta;
        }
        else
        {
            doSaveCurrentTickStateInPrev();
            processedTimeMs += delta;
            return processedTimeMs;
        }
    }

    void BaseTimer::pause()
    {
        if (paused == 0)
        {
            // when pause is 'pressed' first time we force update of current ticks
            getTimeMsInternal(true);
        }
        paused++;
    }

    double BaseTimer::measureTimeResolutionHelper(BaseTimer& tempTimer)
    {
        double result = 0;
        tempTimer.reset();

        do
        {
            result = tempTimer.getTimeMs();
            if (result > DBL_EPSILON) {
                break;
            }
        } while (true);

        return result;
    }

    void BaseTimer::resume()
    {
        if (paused == 0)
            return;

        if (paused == 1)
        {
            // force un pressing of pause
            getTimeMsInternal(true);
        }
        paused--;
    }

    void BaseTimer::reset(double passedmilliseconds)
    {
        processedTimeMs = passedmilliseconds;
        doSaveCurrentTickStateInPrev();
    }

    bool BaseTimer::isPaused() const
    {
        return paused > 0;
    }
}
