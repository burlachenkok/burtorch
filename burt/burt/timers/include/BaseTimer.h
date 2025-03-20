/** @file
* Base class for the timers with different trade-off
*/

#pragma once

#include <string>
#include <sstream>

namespace burt
{
    /** Base class for timers. Supports:
    * - pause state
    * - different accessors for different times
    * - reset timer
    */
    class BaseTimer
    {
    public:
        /** Destructor
        */
        virtual ~BaseTimer() = default;

        /** Local time string
        @return result local time for current timezone
        */
        static ::std::string localTime();

        /** Create timer in not-paused state
        */
        BaseTimer();

        /** Output current elapsed time into text stream and reset the timer
        * @param out text stream used to dump time information
        */
        std::string timeStamp(bool resetTimer = true)
        {
            const auto now = getTimeMs();

            std::stringstream out;
            out << "(Elapsed time: " << now << " msec / stamp #" << stampIndex << ")";
            std::string result = out.str();
            stampIndex += 1;

            if (resetTimer)
                reset();

            return result;
        }

        /** View passed milliseconds in timer (1ms = 1/1000 of second)
        * @return milliseconds until which timer was measuring time
        */
        double getTimeMs();

        /** View passed seconds in timer
        * @return seconds until which timer was measuring time
        */
        double getTimeSec();

        /** Press pause button on "timer"
        */
        void pause();

        /** Resume timer. To make time alive press resume equal to pause times.
        * @remark No effect if time already resume execution
        */
        void resume();

        /** Reset timer to time
        * @param passedmilliseconds timer initialization value in millisecond. Time that passed so far.
        */
        virtual void reset(double passedmilliseconds = 0.0);

        /** Is timer currently not work in the reason because it was paused
        * @return true if timer currently was paused
        */
        bool isPaused() const;

    protected:

        /** Derived timer should support some conception of "previous ticks", and "next ticks", and "number of ticks per second". 
        * @remark Also know as "Template method".
        */
        virtual void doSaveCurrentTickStateInPrev() = 0;

        /** Get delta from last timer tick state. 
        * @remark Also know as "Template method".
        * @return millisecond
        */
        virtual double getDelatMsFromLastTickState() = 0;

        /**
        * Measures the time resolution by utilizing the given temporary timer.
        *
        * @param tempTimer A reference to a temporary timer object used for measuring the time resolution.
        * @return The measured time resolution in milliseconds.
       */
        static double measureTimeResolutionHelper(BaseTimer& tempTimer);

    private:
        double getTimeMsInternal(bool forceUpProcessedTime);

        double processedTimeMs;   ///< Processed time in milliseconds
        int32_t paused;           ///< Count of paused calls
        unsigned int stampIndex;  ///< Timestamp index. Useful to distinguish timestamps.
    };
}
