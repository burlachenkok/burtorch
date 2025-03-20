#include "burt/system/include/threads/Thread.h"
#include "burt/system/include/CpuInfo.h"
#include "burt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <vector>

#include <stdint.h>
#include <math.h>

namespace
{
    int32_t startRoutineReturn123(void* arg, void*) {
        return 123;
    }

    int32_t startRoutineReturnInput(void* arg, void*) {
        burt::DefaultThread::sleepCurrentTh(50);
        return *(static_cast<int32_t*>(arg));
    }
}

TEST(burt, ThreadApiGtest)
{
    {
        int32_t returnCode = 0;
        burt::DefaultThread th1(startRoutineReturn123);
        th1.join();
        EXPECT_TRUE(th1.getExitCode() == 123);

        EXPECT_EQ(th1.isAlive(&returnCode), false);
        EXPECT_EQ(returnCode, 123);
        EXPECT_EQ(th1.isAlive(&returnCode), false);
        EXPECT_EQ(returnCode, 123);
        EXPECT_EQ(th1.isAlive(), false);
    }

    burt::DefaultThread::yeildCurrentTh();
    burt::DefaultThread::yeildCurrentThInHotLoop();
    
    {
        int32_t tmpVar = 9090;

        burt::DefaultThread th2(startRoutineReturnInput, &tmpVar);
        EXPECT_TRUE(th2.isAlive());
        th2.join(1);
        EXPECT_TRUE(th2.isAlive());
        th2.join();
        EXPECT_EQ(th2.isAlive(), false);
    }

    {
        int32_t returnCode = 0;

        burt::DefaultThread thDeferred(startRoutineReturn123, 0, 0, 128);
        thDeferred.join();
        EXPECT_EQ(thDeferred.isAlive(&returnCode), false);
        EXPECT_EQ(returnCode, 123);
        EXPECT_EQ(thDeferred.isAlive(&returnCode), false);
        EXPECT_EQ(returnCode, 123);
        EXPECT_EQ(thDeferred.isAlive(), false);
    }
    burt::DefaultThread::sleepCurrentTh(0);
}

TEST(burt, ThreadApiGtestSleepFor_435_milliseconds)
{
    burt::HighPrecisionTimer tm;
    burt::DefaultThread::sleepCurrentTh(435);
    EXPECT_TRUE(::fabs(435.0 - tm.getTimeMs()) < 200);
    // assume that "defaultTimer+sleep" error is less then 200ms (2/10 of second)

    burt::DefaultThread::yeildCurrentTh();
}

TEST(burt, ThreadsAffinityGTest)
{
#if BURT_MACOS
    std::cout << "There are problems with support thread affinity for MacOS" << '\n';
#else
    if (burt::logicalProcessorsInSystem() < 2)
    {
        std::cout << "In the system there is only processor. Thread Affinity can not be tested.";
    }
    
    burt::DefaultThread::setThreadAffinityMaskForCurrentTh(0x1 << 1);
    EXPECT_TRUE(burt::DefaultThread::getThreadAffinityMaskForCurrentTh() == (0x1 << 1) );

    burt::DefaultThread::setThreadAffinityMaskForCurrentTh(0x1 << 0);
    EXPECT_TRUE(burt::DefaultThread::getThreadAffinityMaskForCurrentTh() == 0x1);    
#endif
}

TEST(burt, ThreadsCheckFlagSetupGTest)
{
    {
        std::atomic<bool> flag = true;
        bool reseted = burt::checkAndResetIfSet(&flag);
        EXPECT_TRUE(reseted == true);
        EXPECT_TRUE(flag == false);
    }

    {
        std::atomic<bool> flag = false;
        bool reseted = burt::checkAndResetIfSet(&flag);
        EXPECT_TRUE(reseted == false);
        EXPECT_TRUE(flag == false);
    }
}
