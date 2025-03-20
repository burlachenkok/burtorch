/** @file
* Thread redirection to platform-specific includes
*/

#pragma once

#if BURT_WINDOWS
    #include "burt/system/include/threads/impl/windows/ThreadWinApi.h"
#elif BURT_LINUX || BURT_MACOS
    #include "burt/system/include/threads/impl/posix/ThreadPosix.h"
#endif

#include <atomic>
#include <assert.h>

namespace burt
{
    #if BURT_WINDOWS
        typedef burt::internal::ThreadWinApi  DefaultThread;
    #elif BURT_LINUX || BURT_MACOS
        typedef burt::internal::ThreadPosix  DefaultThread;
    #endif

        /** Check that boolean flag has been set to "true". If it is, reset it to "false".
        * @return true if flag was set to "true" and reset to "false", false otherwise
        */
        inline bool checkAndResetIfSet(bool* flag)
        {
            if (*flag)
            {
                *flag = false;
                return true;
            }
            else
            {
                return false;
            }
        }

        /** Check that boolean flag has been set to "true". If it is, reset it to "false".
        * @return true if flag was set to "true" and reset to "false", false otherwise
        */
        inline bool checkAndResetIfSet(std::atomic<bool>* flag)
        {
            bool expected = true;
            bool desired = false;

            // this instruction emits implicitly "memory fence" a hardware action that enforces an ordering constraint for instructions
            // - executes atomically
            // - implicit fence
            return std::atomic_compare_exchange_strong(flag, &expected, desired);
            // If false =>  flag (false) != expected (true) => None
            // If  true =>  flag (true) == expected (true) => flag (false) := desired (false)
        }

        inline bool checkAndResetIfSet(volatile bool* flag)
        {
            return burt::internal::myCAS(flag, true, false) == true;
        }
        
        /** Spinlock acquire with atomic CAS operation
        * @see unlockWithAtomic
        */
        inline void lockWithAtomic(std::atomic<bool>* lockVariable)
        {
            bool expected = false;
            bool desired = true;

            // try replace lock to "true", if it was previously "false"
            while (!std::atomic_compare_exchange_strong(lockVariable, &expected, desired))
            {
                // return false => lockForTasks != expected => continue to try
                // warning: C++ API actually modifies expected value
                expected = false;
                DefaultThread::yeildCurrentThInHotLoop();
            }
            assert(*lockVariable == true);
            return;
        }

        /** Spinlock release
        * @see lockWithAtomic
        */
        inline void unlockWithAtomic(std::atomic<bool>* lockVariable) 
        {
            assert(*lockVariable == true);
            *lockVariable = false;
        }


        inline void lockWithAtomic(volatile bool* lockVariable)
        {
            bool expected = false;
            bool desired = true;

            while (!burt::internal::myCAS(lockVariable, expected, desired))
            {
                DefaultThread::yeildCurrentThInHotLoop();
            }
            
            assert(*lockVariable == true);
            return;
        }

        inline void unlockWithAtomic(volatile bool* lockVariable)
        {
            assert(*lockVariable == true);
            *lockVariable = false;
            burt::internal::memoryFence();
        }
        
        
        /** In spinlock is actually locked
        * @return true is spinlock is currently locked, false otherwise
        */
        inline bool isAtomicLocked(std::atomic<bool>* lockVariable)
        {
            return (*lockVariable == true);
        }
}
