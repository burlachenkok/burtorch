/** @file
* Mutex redirection to platform-specific includes
*/

#pragma once

#if BURT_WINDOWS
    #include "burt/system/include/threads/impl/windows/SemaphoreWinApi.h"
#elif BURT_LINUX || BURT_MACOS
    #include "burt/system/include/threads/impl/posix/SemaphorePosix.h"
#endif

namespace burt
{
    #if BURT_WINDOWS
        typedef burt::internal::SemaphoreWinApi DefaultSemaphore;
    #elif BURT_LINUX || BURT_MACOS
        typedef burt::internal::SemaphorePosix DefaultSemaphore;
    #endif
}
