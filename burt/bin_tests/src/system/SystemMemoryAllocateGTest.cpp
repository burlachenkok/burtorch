#include "burt/system/include/SystemMemoryAllocate.h"
#include "burt/system/include/ProcessInfo.h"
#include "burt/system/include/MemInfo.h"

#include "gtest/gtest.h"

TEST(burt, SysMemAllocateGTest)
{
    size_t pageSize = burt::virtualPageSize();

    void* mem = burt::allocateVirtualMemory(pageSize, 1);
    EXPECT_TRUE(mem != nullptr);
    EXPECT_TRUE(burt::lockVirtualMemory(mem, pageSize, 1));
    EXPECT_TRUE(burt::unlockVirtualMemory(mem, pageSize, 1));
    EXPECT_TRUE(burt::deallocateVirtualMemory(mem, pageSize, 1));
    void* memPinned = burt::allocateVirtualMemory(pageSize, 1024);
    EXPECT_TRUE(memPinned != nullptr);

    EXPECT_TRUE(burt::deallocateVirtualMemory(memPinned, pageSize, 1024));

    std::cout << "Information about memory" << '\n';
    std::cout << "  Physical Memory for process: " << burt::physicalMemoryForProcess() / 1024 << " KBytes\n";
    std::cout << "  Virtual and Physical Memory for process: " << burt::totalVirtualAndPhysicalMemoryForProcess() / 1024 << " KBytes\n";
    std::cout << "  Available DRAM memory in the system: " << burt::availablalePhysicalMemoryInBytes() / 1024 / 1024 / 1024 << " GBytes\n";
    std::cout << "  Installed DRAM memory in the system: " << burt::installedPhysicalMemoryInBytes() / 1024 / 1024 / 1024 << " GBytes\n";
    std::cout << '\n';
    std::cout << "  Page Size: " << burt::virtualPageSize()/1024 << " KBytes\n";
    std::cout << "  Process ID: " << burt::currentProcessId() << '\n';
    std::cout << "  Thread ID: " << burt::currentThreadId() << '\n';
    std::cout << '\n';
    burt::ProcessStatistics proc_stats = burt::getProcessStatistics();
    std::cout << "  Memory for dynamic libraries and executable image itself: " << proc_stats.memoryForImages / 1024 << " KBytes\n";
    std::cout << "  Memory for mapped files: " << proc_stats.memoryForMappedFiles / 1024 << " KBytes\n";
    std::cout << "  Private memory allocated for process: " << proc_stats.memoryPrivateForProcess / 1024 << " KBytes\n";
    std::cout << '\n';
}
