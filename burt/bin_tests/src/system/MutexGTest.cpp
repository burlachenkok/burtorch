#include "burt/system/include/threads/Thread.h"
#include "burt/system/include/threads/Mutex.h"
#include "burt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <vector>

#include <stdint.h>

namespace
{
    int32_t startRoutineSleepAndReturn1(void* arg, void*) {
        ((burt::DefaultMutex*)arg)->lock(); // *lock*

        burt::DefaultThread::sleepCurrentTh(50);

        ((burt::DefaultMutex*)arg)->unlock();
        return 1;
    }
}

TEST(burt, MutexApiGTest)
{
    {
        burt::DefaultMutex m1;
        m1.lock();

        EXPECT_EQ(m1.tryLock(), true) << "TRY LOCK ALREADY LOCKED MUTEX";
        m1.unlock();

        m1.unlock();

        m1.lock();
        m1.lock();

        EXPECT_EQ(m1.tryLock(), true) << "TRY LOCK ALREADY LOCKED MUTEX MUTIPLE TIMES BY CURRENT THREAD";
        m1.unlock();

        m1.unlock();
        m1.unlock();
    }

    {
        burt::DefaultMutex m2;
        m2.lock();

        burt::DefaultThread th1(startRoutineSleepAndReturn1, &m2);
        burt::DefaultThread::yeildCurrentThInHotLoop();

        // assume that th1 is in *lock* line of code
        // if comment m2.unlock(); then current thread will be hanged I do not know how to check it via google tests
        m2.unlock();
        th1.join();
    }
}
