#include "SystemMemoryAllocate.h"
#include "burt/system/include/PlatformSpecificMacroses.h"

#include <assert.h>

namespace burt
{
#ifdef BURT_WINDOWS
    void* allocateVirtualMemory(size_t pageSize, size_t numberOfPage)
    {
        // For Windows OS
        //  Regions are always aligned by 64 KBytes
        //  Amount of allocated memory is rounded to page size
        //  https://msdn.microsoft.com/en-us/library/windows/desktop/aa366775(v=vs.85).aspx
        void* ptr = VirtualAlloc(nullptr,
            pageSize * numberOfPage,
            MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        return ptr;
    }

    bool deallocateVirtualMemory(void* regionAddress, size_t pageSize, size_t numberOfPage)
    {
        int res = VirtualFree(regionAddress, 0, MEM_RELEASE);
        return res != 0;
    }

    /* Lock/Pin virtual memory.
    * @param memory pointer to the memory.
    * @param pageSize size of single page.
    * @param numberOfPage number of pages cotained in the memory parameter.
    * @remark All pages in the specified region must be committed. Memory protected with PAGE_NOACCESS cannot be locked.
    * @remark Locking pages into memory may degrade the performance of the system by reducing the available RAM.
    * @remark Pages that a process has locked remain in physical memory until the process unlocks them or terminates.
    * @remark These pages are guaranteed not to be written to the pagefile while they are locked.
    */
    bool lockVirtualMemory(void* memory, size_t pageSize, size_t numberOfPage)
    {
        BOOL result = VirtualLock(memory, pageSize * numberOfPage);
        
        if (result == 0)
        {
            SIZE_T minimumWorkingSetSize = 0;
            SIZE_T maximumWorkingSetSize = 0;
            
            BOOL getWsSize = GetProcessWorkingSetSize(GetCurrentProcess(),
                                                      &minimumWorkingSetSize, 
                                                      &maximumWorkingSetSize);
            // assert(getWsSize != 0);
            
            BOOL setWsSize = SetProcessWorkingSetSize(GetCurrentProcess(), 
                                                      minimumWorkingSetSize, 
                                                      maximumWorkingSetSize + pageSize * numberOfPage);

            // assert(setWsSize != 0);

            result = VirtualLock(memory, pageSize * numberOfPage);
        }
        
        return result != 0;
    }

    bool unlockVirtualMemory(void* memory, size_t pageSize, size_t numberOfPage)
    {
        BOOL result = VirtualUnlock(memory, pageSize * numberOfPage);
        return result != 0;
    }
#else
    void* allocateVirtualMemory(size_t pageSize, size_t numberOfPage)
    {
        // Anonymous mapping.
        void* ptr = mmap(nullptr,
                         pageSize * numberOfPage, // Really kernel allocate memory at page size granularity
                         PROT_READ | PROT_WRITE,  // The content can be read and write
                         MAP_PRIVATE | MAP_ANON,  // Private mapping Modifications are not visible to other processes
                         -1,                      // Values are ignored for anonymous mapping (File handle)
                         0);                      // Values are ignored for anonymous mapping (Offset)

        // The Linux kernel finds a contiguous, unused region in the address space of the application large enough to hold requested bytes.
        //  * mmap() is lazy. It does not immediately allocate physical memory for the requested allocation.
        //  * the first write into such a page causes a page fault.
        
        if (ptr == MAP_FAILED)
        {
            return nullptr;
        }
        else
        {
            return ptr;
        }
    }

    bool deallocateVirtualMemory(void* regionAddress, size_t pageSize, size_t numberOfPages)
    {
        int res = munmap(regionAddress, pageSize * numberOfPages);
        return (res == 0);
    }

    bool lockVirtualMemory(void* memory, size_t pageSize, size_t numberOfPages)
    {
        int res = mlock(memory, pageSize * numberOfPages);
        return (res == 0);
    }

    bool unlockVirtualMemory(void* memory, size_t pageSize, size_t numberOfPages)
    {
        int res = munlock(memory, pageSize * numberOfPages);
        return (res == 0);
    }
#endif
}
