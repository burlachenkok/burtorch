#include "burt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <chrono>
#include <thread>


TEST(burt, HiPrecTimerGTest)
{
    constexpr double kConidenMultiplier = 2.2;
    {
        burt::HighPrecisionTimer tm1;
        EXPECT_TRUE(tm1.getTimeMs() < 10) << "Check init elapsed time for timer";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_GE(tm1.getTimeMs(), 40.0) << "Check lower bound of the elapsed time after sleep";
        EXPECT_LE(tm1.getTimeMs(), kConidenMultiplier * 100.0) << "Check upper bound of the elapsed time after sleep";
        tm1.pause();
        EXPECT_TRUE(tm1.isPaused()) << "Check pause states";
        tm1.pause();
        EXPECT_TRUE(tm1.isPaused()) << "Check that after double press pause button timer will be in paused state";
        tm1.resume();
        EXPECT_TRUE(tm1.isPaused());
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        tm1.resume();

        EXPECT_GE(tm1.getTimeMs(), 40.0);
        EXPECT_LE(tm1.getTimeMs(), kConidenMultiplier * 100.0);
        EXPECT_TRUE(tm1.isPaused() == false);

        tm1.pause();
        EXPECT_GE(tm1.getTimeSec(), 40.0 / 1000.0);
        EXPECT_LE(tm1.getTimeSec(), kConidenMultiplier * 200.0 / 1000.0);

        EXPECT_TRUE(tm1.isPaused());
        tm1.reset(19.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        ASSERT_DOUBLE_EQ(tm1.getTimeMs(), 19.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        ASSERT_DOUBLE_EQ(tm1.getTimeMs(), 19.0);
    }
}

TEST(burt, HiPrecTimerPauseSwitchesGTest)
{
    {
        burt::HighPrecisionTimer tm2;
        EXPECT_TRUE(tm2.isPaused() == false) << "Check that new timer was not in pausable state";
        EXPECT_TRUE(tm2.isPaused() == false) << "Check that new timer was not in pausable state even after duplicate isPaused() calls";

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        tm2.pause();
        EXPECT_TRUE(tm2.isPaused());
        tm2.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_TRUE(tm2.isPaused());
        tm2.resume();

        EXPECT_TRUE(tm2.getTimeMs() < 10) << "Check that 'reset' works in pause state of the timer";
        EXPECT_TRUE(tm2.isPaused() == false);
        for (int i = 0; i < 3; ++i)
            tm2.resume(); // press resume three times

        EXPECT_TRUE(tm2.isPaused() == false);
        tm2.pause();     // press pause one time
        EXPECT_TRUE(tm2.isPaused() == true) << "Check that press 'resume' for alive timer does not have any sense to pause counting";
    }

#if 0
    {
        burt::HighPrecisionTimer tm;
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        double sec = tm.getTimeSec();
        std::cout << sec << " seconds are elapsed\n";
    }
#endif

}

TEST(burt, HiPrecTimerExtraGInfo)
{
    std::cout << "burt::HighPrecisionTimer::measureTimeResolutionInmilliseconds resolution: " << burt::HighPrecisionTimer::measureTimeResolutionInmilliseconds() << " ms" << '\n';
}
