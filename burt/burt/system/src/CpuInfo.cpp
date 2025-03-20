#include "CpuInfo.h"
#include "PlatformSpecificMacroses.h"

#include <iostream>

#if BURT_ARCH_X86_64BIT
    // Add source files in this compilation untit only or x86-64
    #include "burt/3rdparty/vectorclass/physical_processors.cpp"
    #include "burt/3rdparty/vectorclass/instrset_detect.cpp"
    //
    #include "burt/3rdparty/vectorclass/vectorclass.h"
    #include "burt/3rdparty/vectorclass/instrset.h"

    void burt::printExtensionForInstalledCPU(PrintCallBack print)
    {
        const int supported_instructions = instrset_detect();

        if (supported_instructions == 0)
            print("SUPPORTED: 80386 only  |\n");

        if (supported_instructions >= 1)
        {
            print("SUPPORTED: SSE(XMM)    |");
            print("  INFO: 128 bit registers  XMM0 - XMM7.\n");
        }

        if (supported_instructions >= 2)
        {
            print("SUPPORTED: SSE2        |");
            print("  INFO: 128 bit registers  XMM0 - XMM15.\n");
        }
        if (supported_instructions >= 3)
        {
            print("SUPPORTED: SSE3        |");
            print("  INFO: floating point horizontal_add.\n");
        }
        if (supported_instructions >= 4)
        {
            print("SUPPORTED: Suppl. SSE3 |");
            print("  INFO: permute, blend and lookup functions, integer abs.\n");
        }
        if (supported_instructions >= 5)
        {
            print("SUPPORTED: SSE4.1      |");
            print("  INFO: floating point round, truncate, floor, ceil.\n");
        }
        if (supported_instructions >= 6)
        {
            print("SUPPORTED: SSE4.2      |");
            print("  INFO: 64 - bit integer compare 64 bit integer max|min\n");
        }
        if (supported_instructions >= 7)
        {
            print("SUPPORTED: AVX         |");
            print("  INFO: 256 bit registers YMM0 - YMM15 in AMD64.\n");
        }
        if (supported_instructions >= 8)
        {
            print("SUPPORTED: AVX2        |");
            print("  INFO: All operations on 256 - bit integer vectors.\n");
        }
        if (supported_instructions >= 9)
        {
            print("SUPPORTED: AVX512F     |");
            print("  INFO: All operations on 512 - bit integer and floating point vectors ZMM0 - ZMM31\n");
        }
        if (supported_instructions >= 10)
        {
            print("SUPPORTED: AVX512VL    |");
            print("  INFO: Compact boolean vectors for 128 and 256 bit data.\n");

            print("SUPPORTED: AVX512BW    |");
            print("  INFO: 512 bit vectors with 8-bit and 16-bit integer elements.\n");

            print("SUPPORTED: AVX512DQ    |");
            print("  INFO: faster multiplication of vectors of 64 - bit integers.\n");
        }

        if (hasFMA3()) {
            print("SUPPORTED: FMA3        |");
            print("  INFO: floating point code containing multiplication followed by addition\n");
        }

        if (hasFMA4()) {
            print("SUPPORTED: FMA4        |");
            print("  INFO: floating point code containing multiplication followed by addition\n");
        }

        if (hasXOP()) {
            print("SUPPORTED: XOP         |");
            print("  INFO: compare, horizontal_add_x, rotate_left, blend, and lookup.\n");
        }

        if (hasAVX512ER()) {
            print("SUPPORTED: AVX512ER    |");
            print("  INFO: Fast exponential functions.\n");
        }

        if (hasAVX512VBMI()) {
            print("SUPPORTED: AVX512VBMI  |");
            print("  INFO: Faster permutation functions.\n");
        }

        if (hasAVX512VBMI2()) {
            print("SUPPORTED: AVX512VBMI2 |");
            print("  INFO: Faster extract from 8-bit and 16-bit integer vectors.\n");
        }

        // detect if CPU supports the F16C instruction set
        if (hasF16C()) {
            print("SUPPORTED: F16C        |");
            print("  INFO: Conversion between single precisionand half precision floating point numbers.\n");
        }

        // detect if CPU supports the AVX512_FP16 instruction set
        if (hasAVX512FP16()) {
            print("SUPPORTED: AVX512_FP16 instructions |");
            print("  INFO: Half precision floating point calculations\n");
        }
    }
#else

    void burt::printExtensionForInstalledCPU(PrintCallBack print)
    {
        print("Limited information about CPU Extensions for this CPU");
    }
#endif
    
#if BURT_MACOS
    #include <sys/types.h>
    #include <sys/sysctl.h>
#endif

#if BURT_ARCH_ARM
    #if 0
        uint64_t read_mpidr() {
            uint64_t mpidr;
            asm volatile("mrs %0, MPIDR_EL1" : "=r" (mpidr));
            return mpidr;
        }
    #endif
#endif

namespace burt
{
    int physicalProcessorsInSystem()
    {
#if BURT_ARCH_X86_64BIT
        return physicalProcessors();
#else
        // TODO: This is number of logical processors. There is no explicit Posix API to obain information about physical processors
        return sysconf(_SC_NPROCESSORS_CONF);
#endif
    }

    int logicalProcessorsInSystem()
    {
#if BURT_ARCH_X86_64BIT
        int logical_processors = 0;
        physicalProcessors(&logical_processors);
        return logical_processors;
#else
        return sysconf(_SC_NPROCESSORS_CONF);
#endif
    }

    bool getCPUBaseAndMaxFrequencyInMhz(uint64_t& cpuBaseFreqInMhz, uint64_t& maxFreqInMhz)
    {
#if BURT_ARCH_X86_64BIT
        int cpuInfo[4] = { 0, 0, 0, 0 };
        cpuid(cpuInfo, 0);

        const int highestCallingParameter = cpuInfo[0];

        if (highestCallingParameter >= 0x16)
        {
            // https://www.sandpile.org/x86/cpuid.htm#level_0000_0016h
            // https://en.wikipedia.org/wiki/CPUID#:~:text=In%20the%20x86%20architecture%2C%20the,and%20SL%2Denhanced%20486%20processors.
            const int cpuid_cpu_freq_command = 0x16;
            cpuid(cpuInfo, cpuid_cpu_freq_command);
            cpuBaseFreqInMhz = cpuInfo[0];
            maxFreqInMhz = cpuInfo[1];
            return true;
        }
        return false;
#else
        cpuBaseFreqInMhz = maxFreqInMhz = 0;
        return false;
#endif

    }
}

namespace burt
{
#if BURT_MACOS
    int cacheLineSizeForProcessorFromOS()
    {
        uint64_t cacheLineSize = 0;
        int mib[2] = {CTL_HW, HW_CACHELINE};
        size_t length = sizeof(cacheLineSize);
        int res = sysctl(mib, 2, &cacheLineSize, &length, NULL /*new value*/, 0 /*new value len*/);
        assert(res == 0);
        return static_cast<int>(cacheLineSize);
    }

#elif BURT_LINUX
    int cacheLineSizeForProcessorFromOS()
    {
        // L1 data cache line size
        return sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    }

#elif BURT_WINDOWS
    int cacheLineSizeForProcessorFromOS()
    {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = nullptr;
        DWORD returnLength = 0;
        GetLogicalProcessorInformation(buffer, &returnLength);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
        {
            buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength);
            GetLogicalProcessorInformation(buffer, &returnLength);
            
            DWORD byteOffset = 0;
            int lineSize = 0;
            const SYSTEM_LOGICAL_PROCESSOR_INFORMATION* ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)buffer;

            while (lineSize == 0 && byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength)
            {
                switch (ptr->Relationship)
                {
                    case RelationNumaNode:
                        break;
                    case RelationProcessorCore:
                        break;
                    case RelationProcessorPackage:
                        break;

                    case RelationCache:
                        lineSize = ptr->Cache.LineSize;
                        break;
                }
                byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
                ptr++;
            }
            free(buffer);
            return lineSize;
        }
        return 0;
    }
#endif

}

