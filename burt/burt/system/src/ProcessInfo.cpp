#include "ProcessInfo.h"
#include "PlatformSpecificMacroses.h"

#include "burt/system/include/CompilerInfo.h"
#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/system/include/Version.h"
#include "burt/fs/include/StringUtils.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <string>
#include <string_view>

#include <stdint.h>

namespace burt
{
    void printInformationAboutBuild(const char* binaryName)
    {
        std::cout << "About Build Program\n";
        std::cout << "=========================================================================\n";

#if defined(BURT_WINDOWS)
        std::cout << "  Operating System: Windows\n";
#elif defined(BURT_MACOS)
        std::cout << "  Operating System: macOS\n";
#elif defined(BURT_LINUX)
        std::cout << "  Operating System: Linux\n";
#endif
        std::cout << '\n';
        std::cout << "  Binary: " << binaryName << '\n';
        std::cout << "  Project Last Change Date: " << burt::gitlastChangeDate << '\n';
        std::cout << "  Project Last Change id: " << burt::gitLastChange << '\n';
        std::cout << "  Project Branch: " << burt::gitBranch << '\n';
        std::cout << "  Project Build Time: " << burt::buildTimeStamp << '\n';
        std::cout << "  Compiler: " << burt::compilerCppVersion() << '\n';

#if BURT_COMPILER_IS_VC
    #if _DEBUG
        std::cout << "  Build type: DEBUG\n";
    #else
        std::cout << "  Build type: RELEASE\n";
        // https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
    #endif
#else
    #if BURT_DEBUG_BUILD && !BURT_RELEASE_BUILD
            std::cout << "  Build type: DEBUG\n";
    #elif !BURT_DEBUG_BUILD && BURT_RELEASE_BUILD
            std::cout << "  Build type: RELEASE\n";
    #else
            std::cout << "  Build type: UNDEFINED\n";
    #endif
#endif
        std::cout << '\n';
    }

#if BURT_LINUX || BURT_MACOS
    int virtualPageSize()
    {
        return sysconf(_SC_PAGESIZE);
    }

    uint32_t currentProcessId()
    {
        assert(sizeof(getpid()) <= sizeof(uint32_t));
        return (uint32_t)getpid();
    }

    uint64_t currentThreadId()
    {
        assert(sizeof(pthread_self()) <= sizeof(uint64_t));
        return (uint64_t)pthread_self();
    }

    ProcessStatistics getProcessStatistics()
    {
        ProcessStatistics info = {};

        std::stringstream s;
        s << "/proc/" << currentProcessId() << "/maps";

        std::ifstream f(s.str().c_str());

        auto theFieldsDelimiter = [](int c) {
            return c == ' ' || c == '\t';
        };

        auto theAddressSeparator = [](int c) {
            return c == '-';
        };

        if (f.is_open())
        {
            std::string line;
            std::vector<std::string_view> parsed_line;
            std::vector<std::string_view> addresses;

            std::stringstream ss;

            while (std::getline(f, line))
            {
                parsed_line.clear();
                burt::string_utils::splitToSubstrings<false/*return empty strings*/> (parsed_line, line, theFieldsDelimiter);
                assert(parsed_line.size() >= 1);

                //std::cout << line <<  "\n" << parsed_line.size();

                std::string_view addressesString = parsed_line[0];
                addresses.clear();
                burt::string_utils::splitToSubstrings<false/*return empty strings*/>(addresses, addressesString, theAddressSeparator);
                assert(addresses.size() == 2);

                uint64_t startAddress = 0; // Start address of the memory region
                ss << "0x" << addresses[0];
                ss >> std::hex >> startAddress;
                ss.clear();

                uint64_t endAddress = 0;   // End address of the mory region
                ss << "0x" << addresses[1];
                ss >> std::hex >> endAddress;
                ss.clear();

                // The memory region is [startAddress, endAddress)
                // The actual length of memory region in bytes: endAddress - startAddress
                

                uint64_t lengthInBytes = endAddress - startAddress;

                std::string_view path = parsed_line[parsed_line.size() - 1];

                if (parsed_line.size() == 5)
                {
                    // Private memory (in some Linux for heap and tack) -- typically without any tag
                    info.memoryPrivateForProcess += lengthInBytes;
                }
                else if (path.size() >= 3 && path[0] == '/' )
                {
                    size_t so_index = path.rfind(".so");                    // Shared libraries
                    size_t svshm_attach_index = path.rfind("svshm_attach"); // These correspond to the text and data segments of the program

                    if (svshm_attach_index != std::string::npos)
                    {
                        // Private memory
                        //  svshm_attach -- text and data segments of the program
                        info.memoryPrivateForProcess += lengthInBytes;
                    }
                    else if (so_index != std::string::npos)
                    {
                        // Imagies for dynamic Libraries
                        info.memoryForImages += lengthInBytes;
                    }
                    else
                    {
                        // Memory maped files
                        info.memoryForMappedFiles += lengthInBytes;
                    }
                }
                else if (path.find('[') != std::string::npos && path.find(']') != std::string::npos)
                {
                    // Stacks [stack] -- the process stacks
                    // Linux Gate [vdso] -- 
                    info.memoryPrivateForProcess += lengthInBytes;
                }
                else
                {
                    assert(!"NOT PROCESSED CORRECTLY");
                }
            }
            f.close();
        }

        return info;
    }
#elif BURT_WINDOWS
    int virtualPageSize()
    {
        SYSTEM_INFO rawInfo;
        GetSystemInfo(&rawInfo);
        return rawInfo.dwPageSize;
    }

    uint32_t currentProcessId()
    {
        assert(sizeof(GetCurrentProcessId()) <= sizeof(uint32_t));
        return (uint32_t)GetCurrentProcessId();
    }

    uint64_t currentThreadId()
    {
        assert(sizeof(GetCurrentThreadId()) <= sizeof(uint64_t));
        return (uint64_t)GetCurrentThreadId();
    }

    ProcessStatistics getProcessStatistics()
    {
        ProcessStatistics info = {};

        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);

        int MemoryCounter = 0;
        MEMORY_BASIC_INFORMATION    mbi;

        LPVOID lpMem = sysInfo.lpMinimumApplicationAddress;
        while (lpMem < sysInfo.lpMaximumApplicationAddress)
        {
            if (VirtualQuery(lpMem, &mbi, sizeof(mbi)) == 0)
                break;

            //  mbi.RegionSize is the total size(in bytes) of a group of pages that:
            //   - start at mbi.BaseAddress
            //   - have the same security attributes
            //   - have the same state and type

            lpMem = (LPVOID)(((SIZE_T)mbi.BaseAddress) + mbi.RegionSize);

            if (mbi.State == MEM_COMMIT && (mbi.Type & MEM_IMAGE))
            {
                // The memory is reserved and commited for the range of pages
                //  The "image" memory was originally at least has been used do load images of executable entities [dll, exe]
                info.memoryForImages += mbi.RegionSize;
            }
            if (mbi.State == MEM_COMMIT && (mbi.Type & MEM_MAPPED))
            {
                // The memory is reserved and commited for the range of pages
                //  Time "maped" memory has been used as a file mapping object.
                info.memoryForMappedFiles += mbi.RegionSize;
            }
            if (mbi.State == MEM_COMMIT && (mbi.Type & MEM_PRIVATE))
            {
                // The memory is reserved and commited for the range of pages
                //  The "private" memory is backuped in the system paging file.
                info.memoryPrivateForProcess += mbi.RegionSize;
            }
        }

        return info;
    }
#endif
}
