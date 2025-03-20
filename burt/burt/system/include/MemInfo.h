/** @file
* Information about installed CPU
*/

#pragma once

#include <stdint.h>

namespace burt
{
    /** Get total allocated physical memory for process in bytes
    * @return Working set size or Resident set size in bytes
    */
    int64_t physicalMemoryForProcess();

    /** Get total allocated virtual memory for process in bytes. It includes both virtual swapped and resident memory.
    * @return Virtual memory that process is currently used
    * @remark For Windows OS it reports number of actually commited bytes and it does take into reserved addresses. 
    * @remark In Windows OS the virtual memory can be in reseved and in commited states.
    */
    int64_t totalVirtualAndPhysicalMemoryForProcess();
        
    /* Returns the total Available physical DRAM memory in the system
    * @return The amount of actual physical memory, in bytes in System.
    * @remark This is the amount of physical memory that can be immediately reused without having to write its contents to disk first.
    */
    uint64_t availablalePhysicalMemoryInBytes();

    /* Returns the total physical installed DRAM memory in the system
    * @return The amount of actual physical memory, in bytes in System.
    */
    uint64_t installedPhysicalMemoryInBytes();
}
