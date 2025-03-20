#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/system/include/threads/impl/windows/MutexWinApi.h"

#ifdef BURT_WINDOWS

namespace burt
{
    namespace internal
    {

        MutexWinApi::MutexWinApi()
        {
            // InitializeCriticalSectionAndSpinCount for more accurate setup InitializeCriticalSection(&cs, 0x0 - 0x00ffffff);
            InitializeCriticalSection(&cs);
        }

        MutexWinApi::~MutexWinApi()
        {
            DeleteCriticalSection(&cs);
        }

        void MutexWinApi::lock()
        {
            // Wait for HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\CriticalSectionTimeout (30 days)
            EnterCriticalSection(&cs);
        }

        bool MutexWinApi::tryLock()
        {
            if (TryEnterCriticalSection(&cs) == TRUE)
                return true;

            return false;
        }

        void MutexWinApi::unlock()
        {
            LeaveCriticalSection(&cs);
        }
    }
}

#endif
