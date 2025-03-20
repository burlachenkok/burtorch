/** @file
 * C++ check compiler routines
 */

#pragma once

#include <string>
#include <stdint.h>

namespace burt
{
    /* Prin into standart output information about the current program.
    * @param binaryName name of the program
    * @return page size in the platform
    */
    void printInformationAboutBuild(const char* binaryName);

    /* The size of virtual page used for managing data/code for the process
    * @return page size in the platform 
    */
    int virtualPageSize();

    /** Get current process id
    * @return current process id
    */
    uint32_t currentProcessId();

    /** Get current thread id
    * @return current thread id
    */
    uint64_t currentThreadId();

    struct ProcessStatistics
    {
        int64_t memoryForImages;            ///< Memory for dynamic libraries and executable image itself
        int64_t memoryForMappedFiles;       ///< Memory for mapped files
        int64_t memoryPrivateForProcess;    ///< Private memory allocated for process
    };

    /** Get process system statistic info about committed memory. This memory is either is allocated in DRAM or is in the Swap File.
    * @return structure with statistic info per process
    */
    ProcessStatistics getProcessStatistics();
}
