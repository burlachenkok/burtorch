/** @file
* Mutex redirection to platform-specific includes
*/

#pragma once

#if BURT_WINDOWS
    #include "burt/system/include/threads/impl/windows/MutexWinApi.h"
#elif BURT_LINUX || BURT_MACOS
    #include "burt/system/include/threads/impl/posix/MutexPosix.h"
#endif

namespace burt
{
    #if BURT_WINDOWS
        typedef burt::internal::MutexWinApi DefaultMutex;
    #elif BURT_LINUX || BURT_MACOS
        typedef burt::internal::MutexPosix DefaultMutex;
    #endif
}
