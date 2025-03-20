/** @file
* Threading support based on support from Posix OS standard
*/

#pragma once

#if BURT_LINUX || BURT_MACOS

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/system/include/threads/Mutex.h"

#include <pthread.h>

#include <atomic>

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

namespace burt
{
    namespace internal
    {
        /** Garantee that all write are written to L1 Cache and there is no read ahead
        */
        inline void memoryFence() {
            __sync_synchronize();
        }

        /** Performs atomic compare and swap operation
        * @param dest destination address
        * @param expected expected value in *dest
        * @param desired value which should be placed in *dest if [*dest == expected]
        * @return value in *dest has been replaced with desired because CAS test has been succeeded
        */
        template <class T>
        inline bool myCAS(volatile T* dest, T expected, T desired)
        {
            if constexpr (sizeof(T) * 8 == 8)
            {

                bool oldDestinationIsReplaced = __sync_bool_compare_and_swap(reinterpret_cast<volatile int8_t*>(dest),
                                                                             reinterpret_cast<int8_t&>(expected),
                                                                             reinterpret_cast<int8_t&>(desired));
                return oldDestinationIsReplaced;
            }
            else if constexpr (sizeof(T) * 8 == 16)
            {
                assert(intptr_t(dest) % 2 == 0); // Ddest should be aligned by 2 bytes
                
                bool oldDestinationIsReplaced = __sync_bool_compare_and_swap(reinterpret_cast<volatile int16_t*>(dest),
                                                                             reinterpret_cast<int16_t&>(expected),
                                                                             reinterpret_cast<int16_t&>(desired));
                return oldDestinationIsReplaced;
            }
            else if constexpr (sizeof(T) * 8 == 32)
            {
                assert(intptr_t(dest) % 4 == 0); // Ddest should be aligned by 4 bytes
                bool oldDestinationIsReplaced = __sync_bool_compare_and_swap(reinterpret_cast<volatile int32_t*>(dest),
                                                                             reinterpret_cast<int32_t&>(expected),
                                                                             reinterpret_cast<int32_t&>(desired));
                return oldDestinationIsReplaced;
            }
            else if constexpr (sizeof(T) * 8 == 64)
            {
                assert(intptr_t(dest) % 8 == 0); // Ddest should be aligned by 8 bytes

                bool oldDestinationIsReplaced = __sync_bool_compare_and_swap(reinterpret_cast<volatile int64_t*>(dest),
                                                                             reinterpret_cast<int64_t&>(expected),
                                                                             reinterpret_cast<int64_t&>(desired));
                                                                     
                return oldDestinationIsReplaced;
            }
            else
            {
                assert(!"UNLIKELY CASE. BACKUP IMPLEMENTATION. PRETTY BAD FOR PERFORMANCE.");

                static burt::DefaultMutex m;
                m.lock();

                T oldDestination = *dest;
                if (oldDestination == expected)
                {
                    *dest = desired;
                }
                m.unlock();

                return oldDestination == expected;
            }
        }
        
        /** Threading based on PThread support.
        */
        class ThreadPosix
        {
        public:
            typedef int32_t (*StartRoutine)(void* arg1, void* arg2);

            /** Create thread in current process
            * @param start start routine
            * @param userArg extra arg to to thread startup code
            * @param createDeferred not create real thread instead the function will be evaluated after call join or isAlive
            */
            explicit ThreadPosix(StartRoutine start, void* userArg1 = 0, void* userArg2 = 0, size_t reserverStackSize = 0);

            /** Destructor.
            */
            ~ThreadPosix();

            ThreadPosix(const ThreadPosix&) = delete;

            ThreadPosix(ThreadPosix&&) = delete;

            /** Is thread alive (continue to execute)
            * @param returnCode is non zero the return code will be place here
            * @return true - if thread is still alive
            */
            bool isAlive(int32_t* returnCode = 0) const;

            /** Wait for end of the thread until milliseconds
            * @param milliseconds wait timeout
            */
            void join(uint32_t milliseconds);

            /** Wait until the thread will finished its work
            */
            void join();

            /** Get exit code of the terminated thread
            */
            int32_t getExitCode();

            /** Suspend thread execution
            * @sa resume
            */
            void suspend();

            /** Resume thread execution
            * @sa suspend
            */
            void resume();

            /** Sleep current thread until timeout ms
            * @param milliseconds timeout
            */
            static void sleepCurrentTh(uint32_t milliseconds);

            /** Nothing todo in current thread. Give execution to another thread.
            */
            static void yeildCurrentTh();

            /** Nothing todo in current thread. Optionally give execution to another thread.
            */
            static void yeildCurrentThInHotLoop();

            /** Setup hard affinity (or binding mask) for current executed CPU thread.
            * This call will allow execution of the thread only on specific processors
            * @param mask mask which contain bitwise "1" in corresponding position in which processors it's possible to execute thread
            */
            static void setThreadAffinityMaskForCurrentTh(uint64_t mask);

            /** Get hard affinity (or binding mask) for current executed CPU thread.
            * @return current affinity mask for specific thread
            */
            static uint64_t getThreadAffinityMaskForCurrentTh();

            /** Setup hard affinity (or binding mask) for current executed CPU thread.
            * This call will allow execution of the thread only on specific processors
            * @param mask mask which contain bitwise "1" in corresponding position in which processors it's possible to execute thread
            */
            void setThreadAffinityMask(uint64_t mask);

            /** Get hard affinity (or binding mask) for current executed CPU thread.
            * @return current affinity mask for specific thread
            */
            uint64_t getThreadAffinityMask();
            
        private:
            StartRoutine start;
            void* userArg1;
            void* userArg2;

            std::atomic<int32_t> resultCodeAttr;
            std::atomic<int32_t> isAliveAttr;

            pthread_t workerThread;
            bool workerThreadHasBeenJoined;
            
            friend void* InternalWorkerThreadPosix(void *param);
        };
    }
}

#endif
