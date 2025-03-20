#include "PlatformSpecificMacroses.h"
#include "burt/system/include/threads/impl/windows/SemaphoreWinApi.h"
#include <assert.h>

#ifdef BURT_WINDOWS

namespace burt 
{
    namespace internal 
    {
        SemaphoreWinApi::SemaphoreWinApi()
        : semaphore(HANDLE())
        {}
        
        SemaphoreWinApi::SemaphoreWinApi(uint32_t theInitialCount)
        : semaphore(HANDLE())
        {
            semaphore = ::CreateSemaphore(NULL, theInitialCount, LONG_MAX, 0);
            assert(semaphore != NULL && semaphore != INVALID_HANDLE_VALUE);
        }
        
        SemaphoreWinApi::SemaphoreWinApi(const SemaphoreWinApi& rhs)
        : semaphore(HANDLE())
        {
            ::DuplicateHandle(GetCurrentProcess(), rhs.semaphore, GetCurrentProcess(), &semaphore, 0, false, DUPLICATE_SAME_ACCESS);
        }


        SemaphoreWinApi::SemaphoreWinApi(SemaphoreWinApi&& rhs) noexcept
        {
            semaphore = rhs.semaphore;
            rhs.semaphore = HANDLE();
        }

        SemaphoreWinApi& SemaphoreWinApi::operator = (SemaphoreWinApi&& rhs) noexcept
        {
            if (this == &rhs)
                return *this;
            
            if (semaphore != HANDLE()){
                ::CloseHandle(semaphore);
            }

            semaphore = rhs.semaphore;
            rhs.semaphore = HANDLE();
            return *this;
        }

        SemaphoreWinApi::~SemaphoreWinApi()
        {
            if (semaphore != HANDLE())
            {
                ::CloseHandle(semaphore);
            }
        }

        void SemaphoreWinApi::acquire()
        {
            ::WaitForSingleObject(semaphore, INFINITE);
        }

        bool SemaphoreWinApi::tryAcquire()
        {
            return ::WaitForSingleObject(semaphore, 0) != WAIT_TIMEOUT;

        }

        void SemaphoreWinApi::release(int32_t count)
        {
            ::ReleaseSemaphore(semaphore, count, 0);
        }
    }
}
#endif
