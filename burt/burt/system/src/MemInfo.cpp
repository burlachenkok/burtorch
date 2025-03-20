#include "MemInfo.h"
#include "PlatformSpecificMacroses.h"

#include "burt/fs/include/StringUtils.h"
#include "burt/system/include/ProcessInfo.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include <assert.h>
#include <string.h>

#if BURT_MACOS
    #include <mach/mach.h>
    #include <mach/vm_statistics.h>

    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <unistd.h>

    #include <libproc.h>
    #include <sys/proc_info.h>
#endif

namespace
{
    int64_t getProcessStatus(int pid, const char* valueName)
    {
        std::stringstream s;
        s << "/proc/" << pid << "/status";
        
        std::ifstream f(s.str().c_str());
        size_t valueNameLength = strlen(valueName);

        if (f.is_open()) 
        {
            std::string line;
            while (std::getline(f, line)) 
            {
                size_t valuePos = line.find(valueName);
                if (valuePos == std::string::npos)
                    continue;

                size_t iStart = valuePos + valueNameLength;

                for (;iStart < line.size(); ++iStart)
                {
                    if (isdigit(line[iStart]))
                        break;
                }

                size_t iEnd = iStart + 1;

                for (;iEnd < line.size(); ++iEnd)
                {
                    if (!isdigit(line[iEnd]))
                        break;
                }

                // Sequence in [iStart, iEnd) contain the string value
                uint64_t value = 0;
                bool valueIsOK = burt::string_utils::fromStringUnsignedInteger(value,
                                                                               line.data() + iStart,
                                                                               line.data() + iEnd);
                assert(valueIsOK);

                // Collect information about prefix
                std::stringstream prefix;

                for (size_t i = iEnd; i < line.size(); ++i)
                {
                    if (isspace(line[i]))
                        continue;
                    else
                    {
                        prefix << char(tolower(line[i]));
                    }
                }

                if (prefix.str() == "kb")
                    value *= 1024;
                else if (prefix.str() == "mb")
                    value *= (1024 * 1024);

                return value;
            }
            
            f.close();
        }
        
        return 0;
    }
}

namespace burt
{

    int64_t physicalMemoryForProcess()
    {
#ifdef BURT_WINDOWS
        PROCESS_MEMORY_COUNTERS_EX ppmc;
        
        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*) &ppmc, sizeof(ppmc)) == TRUE)
        {
            return ppmc.WorkingSetSize;
        }
        else
        {
            assert(!"ISSUES WITH GATHER WORKING SET SIZE");
            return 0;
        }
#elif BURT_LINUX
      return getProcessStatus(getpid(), "VmRSS");
#elif BURT_MACOS
        proc_taskinfo task_info;
        if (proc_pidinfo(getpid(), PROC_PIDTASKINFO, 0, &task_info, sizeof(task_info)) == -1)
            return 0;
        return task_info.pti_resident_size;
#else
    #error "Uknown OS"
#endif
    }

    int64_t totalVirtualAndPhysicalMemoryForProcess()
    {
#ifdef BURT_WINDOWS
        PROCESS_MEMORY_COUNTERS_EX ppmc;

        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&ppmc, sizeof(ppmc)) == TRUE)
        {
            return ppmc.PrivateUsage;
        }
        else
        {
            assert(!"ISSUES WITH GATHER WORKING SET SIZE");
            return 0;
        }
#elif BURT_LINUX
        return getProcessStatus(getpid(), "VmSize");
#elif BURT_MACOS
        proc_taskinfo task_info;
        if (proc_pidinfo(getpid(), PROC_PIDTASKINFO, 0, &task_info, sizeof(task_info)) == -1)
            return 0;
        return task_info.pti_virtual_size;
#else
        #error "Unknown OS"
#endif
    }
    
    uint64_t availablalePhysicalMemoryInBytes()
    {
        uint64_t physicalMemoryTotal = 0;

#if BURT_WINDOWS
        // 8.5, 8.5.1 Initializer from C++2003-10-15.
        // If, for an array or structure, the number of initialization values is less than the dimension of the array
        // or structure, the rest of the elements are initialized with the default value for the corresponding type
        // // (as in the case of initialization of static variables
        MEMORYSTATUSEX meminfo = {};

        meminfo.dwLength = sizeof(meminfo);
        GlobalMemoryStatusEx(&meminfo);
        physicalMemoryTotal = meminfo.ullAvailPhys;
#elif BURT_LINUX
        struct sysinfo meminfo;
        sysinfo(&meminfo);
        
        physicalMemoryTotal = meminfo.freeram;
        physicalMemoryTotal *= meminfo.mem_unit;
#else
        // This is pretty poor documented for Darwin, but if combine this sources:
        //   https://blog.guillaume-gomez.fr/articles/2021-09-06+sysinfo%3A+how+to+extract+systems%27+information
        //   https://gist.github.com/sck/3439836
        //   https://stackoverflow.com/questions/14789672/why-does-host-statistics64-return-inconsistent-results
        //   https://web.mit.edu/darwin/src/modules/xnu/osfmk/man/vm_statistics.html
        //   https://apple.stackexchange.com/questions/347037/what-is-the-purpose-of-speculative-memory#:~:text=Speculative%20memory%20(introduced%20with%20OS,it%20hasn't%20happened%20yet.

        uint64_t totalInstalledRam = installedPhysicalMemoryInBytes();

        vm_statistics64 vm_st;
        mach_port_t host    = mach_host_self();
        natural_t   count   = HOST_VM_INFO64_COUNT;
        kern_return_t res = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_st, &count);
        assert(res == KERN_SUCCESS);

        physicalMemoryTotal = vm_st.free_count * burt::virtualPageSize();

#if 0
        physicalMemoryTotal = totalInstalledRam -
                                (vm_st.active_count +      /* [pageable] The total number of pages currently in use and pageable.*/
                                 vm_st.inactive_count +    /* [probably in DRAM] The number of inactive pages.*/
                                 vm_st.wire_count +        /* [pinned pages] The number of pages that are wired in memory and cannot be paged out. */
                                 vm_st.speculative_count - /* [?] Kernel predicts that it probably will be used for something later, but it hasn't happened yet. */
                                 vm_st.purgeable_count)    /* not needed anymore pages */
                                * burt::virtualPageSize();
#endif

#endif
        return physicalMemoryTotal;
    }

    uint64_t installedPhysicalMemoryInBytes()
    {
        uint64_t physicalMemoryTotal = 0;

#if BURT_WINDOWS
        MEMORYSTATUSEX meminfo = {};
        meminfo.dwLength = sizeof(meminfo);
        GlobalMemoryStatusEx(&meminfo);
        physicalMemoryTotal = meminfo.ullTotalPhys;
#elif BURT_LINUX
        struct sysinfo meminfo = {};
        sysinfo(&meminfo);

        physicalMemoryTotal = meminfo.totalram;
        physicalMemoryTotal *= meminfo.mem_unit;
#elif BURT_MACOS
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        size_t length = sizeof(physicalMemoryTotal);
        int res = sysctl(mib, 2, &physicalMemoryTotal, &length, NULL /*new value*/, 0 /*new value len*/);
        assert(res == 0);
#else
        #error "Uknown OS"
#endif

        return physicalMemoryTotal;
    }
}
