#include "burt/system/include/threads/impl/posix/MutexPosix.h"

#if BURT_LINUX || BURT_MACOS

namespace burt
{
    namespace internal
    {

        MutexPosix::MutexPosix()
        {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
            pthread_mutex_init(&locker, &attr);
        }

        MutexPosix::~MutexPosix()
        {
            pthread_mutex_destroy(&locker);
        }

        void MutexPosix::lock()
        {
            pthread_mutex_lock(&locker);
        }

        bool MutexPosix::tryLock()
        {
            if (pthread_mutex_trylock(&locker) == 0)
            {
                return true;
            }

            return false;
        }

        void MutexPosix::unlock()
        {
            pthread_mutex_unlock(&locker);
        }
    }
}
#endif
