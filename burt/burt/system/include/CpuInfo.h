/** @file
* Information about installed CPU
*/

#pragma once

#include <stdint.h>

namespace burt
{
    /** Output information about available cpu extensions
    */
    typedef void (*PrintCallBack)(const char* msg);
    
    /** Print information about properties of installed CPU in the system.
    * @param print callback responsible for printing single string
    */
    void printExtensionForInstalledCPU(PrintCallBack print);

    /* Returns the number of physical processors, i.e. the number of CPU cores.
    * @return number of processors
    */
    int physicalProcessorsInSystem();

    /* Returns the number of logical processors, i.e. the number of CPU cores.
    * This is called simultaneous multi-threading (SMT) or hyper-threading.
    * @return number of processors
    * @remark Two threads running in the same CPU core will be competing for the same CPU resources. SMT is not advantageous for CPU-intensive code.
    */
    int logicalProcessorsInSystem();

    /** Cache line - is size of bytes real fetching by processor from memory when it need even one byte. Aspects:
    * 1. Also for optimization purposes group similar access data to cache line. If write one byte to memory then it will be written full cache line.
    * 2. Linear array traversals very cache-friendly.
    * 3. Cache Coherency was supported automatically by CPU.
    * 4. False sharing - Different cores concurrently access same cache line and at least one is a writer
    * @return This is the size of L1 data cache line in bytes from OS
    */
    int cacheLineSizeForProcessorFromOS();

    /** Cache line - is size of bytes real fetching by processor from memory when it need even one byte. Aspects:
    * 1. Also for optimization purposes group similar access data to cache line. If write one byte to memory then it will be written full cache line.
    * 2. Linear array traversals very cache-friendly.
    * 3. Cache Coherency was supported automatically by CPU.
    * 4. False sharing - Different cores concurrently access same cache line and at least one is a writer
    * @return the size of cache line in bytes
    * @remark Authors of C++17 standard claims that these constants provide a portable way to access the L1 data cache line size.
    */
    inline int cacheLineSizeForProcessorFromCpp()
    {
#if __cpp_lib_hardware_interference_size >= 201703L
        // These constants provide a portable way to access the L1 data cache line size.
        return std::hardware_destructive_interference_size;
#else
        return cacheLineSizeForProcessorFromOS();
#endif
    }

    /** Get for install CPU core processor base frequency and maximum frequency
    * @param[out] cpuBaseFreq base frequency for CPU core
    * @param[out] cpuMaxFreq maximum frequency for CPU core
    * @return true if cpuBaseFreq and cpuMaxFreq has been filled
    */
    bool getCPUBaseAndMaxFrequencyInMhz(uint64_t& cpuBaseFreqInMhz, uint64_t& maxFreqInMhz);
}
