/** @file
* C++ check compiler routines
*/

#pragma once

#include <stddef.h>

namespace burt
{
    void* allocateVirtualMemory(size_t pageSize, size_t numberOfPage);
    
    bool deallocateVirtualMemory(void* regionAddress, size_t pageSize, size_t numberOfPage);

    bool lockVirtualMemory(void* memory, size_t pageSize, size_t numberOfPage);

    bool unlockVirtualMemory(void* memory, size_t pageSize, size_t numberOfPage);
}
