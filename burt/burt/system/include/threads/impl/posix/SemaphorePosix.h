/** @file
* Semaphore, kernel-object
*/



#pragma once

#if BURT_LINUX || BURT_MACOS

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/system/include/threads/Mutex.h"

#include <pthread.h>
#include <semaphore.h>
#include <atomic>

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

namespace burt
{
    namespace internal
    {
        /** Semaphore impl based on WinApi implementation.
        */
        class SemaphorePosix
        {
        public:
            SemaphorePosix();

            SemaphorePosix(uint32_t theInitialCount);

            SemaphorePosix(const SemaphorePosix& rhs) = delete;

            SemaphorePosix(SemaphorePosix&& rhs) noexcept;

            SemaphorePosix& operator = (SemaphorePosix&& rhs) noexcept;

            ~SemaphorePosix();

            void acquire();

            bool tryAcquire();

            void release(int32_t count);

        private:
            bool semIsValid;
            sem_t sem;
        };
    }
}

#endif
