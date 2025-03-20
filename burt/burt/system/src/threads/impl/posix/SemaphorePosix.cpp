#include "PlatformSpecificMacroses.h"
#include "burt/system/include/threads/impl/posix/SemaphorePosix.h"

#include <assert.h>

#if BURT_LINUX || BURT_MACOS

#include <semaphore.h>
#include <errno.h>

namespace burt
{
    namespace internal
    {
        SemaphorePosix::SemaphorePosix()
        : sem()
        , semIsValid(false)
        {}

        SemaphorePosix::SemaphorePosix(uint32_t theInitialCount)
        : sem()
        , semIsValid(false)
        {
            int res = sem_init(&sem,
                               0,                             // semaphore is to be shared between threads of current process
                               (unsigned int) theInitialCount // initial value
                               );
            if (res != -1)
            {
                semIsValid = true;
            }

            assert(res != -1);
        }

        SemaphorePosix::SemaphorePosix(SemaphorePosix&& rhs) noexcept
        {
            sem = rhs.sem;
            semIsValid = rhs.semIsValid;
            
            rhs.semIsValid = false;
        }

        SemaphorePosix& SemaphorePosix::operator = (SemaphorePosix&& rhs) noexcept
        {
            if (semIsValid) {
                sem_destroy(&sem);
            }
            sem = rhs.sem;
            semIsValid = rhs.semIsValid;
            
            rhs.semIsValid = false;

            return *this;
        }

        SemaphorePosix::~SemaphorePosix()
        {
            if (semIsValid) {
                sem_destroy(&sem);
            }
        }

        void SemaphorePosix::acquire()
        {
            int res = sem_wait(&sem);
            assert(res != -1);
        }

        bool SemaphorePosix::tryAcquire()
        {
            int res = sem_trywait(&sem);
            if (res == EAGAIN)
                return  false;

            assert(res != -1);
            return true;
        }

        void SemaphorePosix::release(int32_t count)
        {
            for (;count > 0; count--)
            {
                int res = sem_post(&sem);
                assert(res != -1);
            }
        }
    }
}

#endif
