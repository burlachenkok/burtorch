#include "burt/system/include/threads/impl/posix/ThreadPosix.h"
#include "burt/timers/include/HighPrecisionTimer.h"

#if BURT_LINUX || BURT_MACOS

#include <sched.h>
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <assert.h>

namespace burt
{
    namespace internal
    {
        void* InternalWorkerThreadPosix(void *param)
        {
            ThreadPosix* thContext = static_cast<ThreadPosix*>(param);
            thContext->resultCodeAttr = thContext->start(thContext->userArg1, thContext->userArg2);
            thContext->isAliveAttr = false;
            return nullptr;
        }

        ThreadPosix::ThreadPosix(ThreadPosix::StartRoutine theStart, void* theUserArg1, void* theUserArg2, size_t reserverStackSize)
        : start(theStart)
        , userArg1(theUserArg1)
        , userArg2(theUserArg2)
        , resultCodeAttr(0)
        , isAliveAttr(true)
        , workerThreadHasBeenJoined(false)
        {
            // create thread with default attributes
            //  pthread_attr_t attr;
            //  pthread_attr_setstacksize(&attr, reserverStackSize);
            //
            // By default: pthread_join creates a thread in joinable state
            //
            // By default in Linux: 2MBytes / stack per thread (and it seems there is nothing like GUARD_PAGE as in Windows)
            //
            int thHasBeenCreated = pthread_create(&workerThread, nullptr, InternalWorkerThreadPosix, this);
            
            assert(thHasBeenCreated == 0);

            if (thHasBeenCreated != 0)
            {
                std::cerr << "Posix: Problems with creating one more thread.\n";
                isAliveAttr = false;
                abort();
            }            
        }

        ThreadPosix::~ThreadPosix()
        {
            if (isAlive())
            {
                std::cerr << "WARNING: You're trying to call destructor for not completed thread!\n";
            }
            
            if (workerThreadHasBeenJoined == false)
            {
                // After pthread_detach it would be not possible to:
                //  join or obtain return status

                // Threads (posix) can be either:
                // - joinable [by default]
                // - detached 
                pthread_detach(workerThread);
            }
        }

        bool ThreadPosix::isAlive(int32_t* returnCode) const
        {
            if (isAliveAttr)
                return true;
            else
            {
                if (returnCode)
                    *returnCode = resultCodeAttr;
                return false;
            }
        }

        void ThreadPosix::suspend()
        {
            assert(!"NOT IMPLEMENTED IN POSIX");
        }

        void ThreadPosix::resume()
        {
            assert(!"NOT IMPLEMENTED IN POSIX");
        }

        void ThreadPosix::join(uint32_t milliseconds)
        {
            // join already joined thread lead to undefined behavior in POSIX
            assert(workerThreadHasBeenJoined == false);

#if BURT_LINUX
            
            timespec timeout;
            timeout.tv_sec = milliseconds / 1000;
            timeout.tv_nsec = (milliseconds % 1000) * 1000 * 1000;

            int joinReturnValue = pthread_timedjoin_np(workerThread, nullptr, &timeout);

            if (joinReturnValue == 0)
            {
                // workerThreadHasBeenJoined = true;
                return;                
            }
            else if (joinReturnValue == ETIMEDOUT)
            {
                return;
            }
            else
            {
                printf("%s\n", strerror(joinReturnValue));
                assert(!"POSIX pthread_timedjoin_np induced error");

                return;
            }
            
#elif BURT_MACOS
            
            burt::HighPrecisionTimer timer;
            
            for (;;)
            {
                if (isAliveAttr == false)
                {
                    // Thread has finished his work
                    break;
                }
                else if (timer.getTimeMs() >= milliseconds)
                {
                    // Time is up
                    break;
                }
                else
                {
                    // Time is not up yet and according to isAliveAttr thread is still alive
                    yeildCurrentTh();
                }                
            }
            return;
#else
            
            #error "Please specify logic for ThreadPosix::join(uint32_t milliseconds)"
#endif
        }

        void ThreadPosix::join()
        {
            // join already joined thread lead to undefined behavior in POSIX
            assert(workerThreadHasBeenJoined == false); 
            
            // join the thread execution (wait) and ignoring return value
            workerThreadHasBeenJoined = true;

            // if multiple threads called pthread_join -- result is undefined
            int joinReturnValue = pthread_join(workerThread, nullptr);

            if (joinReturnValue == ESRCH)
            {
                // Either:
                //   a. It is a bug in this/clients code
                //   b. The thread has actually finished his work
                return;
            }
            else
            {
                assert(joinReturnValue == 0);
                return;
            }
        }

        int32_t ThreadPosix::getExitCode() {
            return resultCodeAttr;
        }

        void ThreadPosix::sleepCurrentTh(uint32_t milliseconds)
        {
            ::usleep(milliseconds * 1000);
        }

        void ThreadPosix::yeildCurrentTh()
        {
#if BURT_LINUX
                pthread_yield();
#elif BURT_MACOS
                pthread_yield_np();
#else
                #error "Please specify logic for ThreadPosix::yeild()"
#endif
        }
        
        void ThreadPosix::yeildCurrentThInHotLoop()
        {
#if 0
            yeildCurrentTh();
#else
            constexpr uint16_t kSpinCounter = 2000;
            
            static thread_local uint16_t spinCounter4CurrentThread = 0;
            spinCounter4CurrentThread++;

            if (spinCounter4CurrentThread == kSpinCounter)
            {
                spinCounter4CurrentThread = 0;
                yeildCurrentTh();
            }
#endif
        }

        void ThreadPosix::setThreadAffinityMaskForCurrentTh(uint64_t mask)
        {
#if BURT_LINUX
            pthread_t thread = pthread_self();

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);

            for (size_t j = 0; j < 64; j++)
            {
                if (mask & (0x1 << j))
                {
                    CPU_SET(j, &cpuset);
                }
            }
            
            // https://man7.org/linux/man-pages/man3/pthread_getaffinity_np.3.html
            int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
            
            assert(result == 0);
#elif BURT_MACOS
            assert(!"Affinity support has problems for macOS");
            return;
#endif
        }

        uint64_t ThreadPosix::getThreadAffinityMaskForCurrentTh()
        {
#if BURT_LINUX
            uint64_t result = 0;

            pthread_t thread = pthread_self();
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);

            /* Check the actual affinity mask assigned to the thread */
            int s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
            assert(s == 0);

            for (size_t j = 0; j < CPU_SETSIZE; j++)
            {
                if (CPU_ISSET(j, &cpuset))
                {
                    result |= (0x1 << j);
                }
            }

            return result;
#else
            // Affinity support has problems for macOS
            return 0;
#endif
        }

        void ThreadPosix::setThreadAffinityMask(uint64_t mask)
        {
#if BURT_LINUX
            pthread_t thread = workerThread;

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);

            for (size_t j = 0; j < 64; j++)
            {
                if (mask & (0x1 << j))
                {
                    CPU_SET(j, &cpuset);
                }
            }

            // https://man7.org/linux/man-pages/man3/pthread_getaffinity_np.3.html
            int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

            assert(result == 0);
#elif BURT_MACOS
            // Affinity support has problems for macOS
            return;
#endif
        }

        uint64_t ThreadPosix::getThreadAffinityMask()
        {
#if BURT_LINUX
            uint64_t result = 0;
            
            pthread_t thread = workerThread;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);

            /* Check the actual affinity mask assigned to the thread */
            int s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
            assert(s == 0);
            
            for (size_t j = 0; j < CPU_SETSIZE; j++)
            {
                if (CPU_ISSET(j, &cpuset))
                {
                    result |= (0x1 << j);                    
                }
            }

            return result;
#else
            // Affinity support has problems for macOS
            return 0;
#endif
        }
    }
}

#endif
