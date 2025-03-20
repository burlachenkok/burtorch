/** @file
* Threading support based on native WinAPI
*/

#pragma once

#if BURT_WINDOWS

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/system/include/threads/Mutex.h"

#include <atomic>

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

namespace burt
{
    namespace internal
    {
        /** Garantee that all write are written to L1 Cache and there is no read ahead
        */
        inline void memoryFence()
        {
            MemoryBarrier();
        }

       /** Performs atomic compare and swap operation
       * @param dest destination address
       * @param expected expected value in *dest
       * @param desired value which should be placed in *dest if [*dest == expected]
       * @return value in *dest has been replaced with desired because CAS test has been succeeded
       * @remark In x86 cmpxcg exectutes in 50 cycles, while move to kernel space costs 1000 cycles.
       * @remark In case of using Compare-And-Swap(CAS) or another atomic write operations Harware garantees complete update of the value.
       */
       template <class T>
       inline bool myCAS(volatile T* dest, T expected, T desired)
       {
#if 0
           if constexpr (0)
           {
               // FOR DEBUGGING
               static burt::DefaultMutex m;
               m.lock();

               T oldDestination = *dest;
               if (oldDestination == expected)
               {
                   *dest = desired;
               }
               m.unlock();
               return oldDestination == expected;
           }
#endif

           if constexpr (sizeof(T) * 8 == 8)
           {
               // A Bit hacky, but Windows API does not support InterlockedCompareExchange8
               // Implementation below is independent of Endiadness of CPU
               using dtype = SHORT;
               static_assert(sizeof(dtype) == 2);

               dtype s_desired, s_expected;
               
               // get raw byte view
               unsigned char* b_desired = reinterpret_cast<unsigned char*>(&s_desired);
               unsigned char* b_expected = reinterpret_cast<unsigned char*>(&s_expected);
               
               b_desired[0]  = reinterpret_cast<const unsigned char&>(desired);
               b_expected[0] = reinterpret_cast<const unsigned char&>(expected);

               // get next byte from memory and put into b_desired, and b_expected
               b_desired[1] = b_expected[1] = (reinterpret_cast<volatile unsigned char*>(dest))[1];
               
               dtype oldDestination = InterlockedCompareExchange16(reinterpret_cast<volatile dtype*>(dest), s_desired, s_expected);
               return oldDestination == s_expected;
           }
           else if constexpr (sizeof(T) * 8 == 16)
           {
               // SHORT -- 16-bit signed integer
               using dtype = SHORT;
               assert(intptr_t(dest) % sizeof(dtype) == 0); // Ddest should be aligned by 2 bytes
               dtype oldDestination = ::InterlockedCompareExchange16(reinterpret_cast<volatile dtype*>(dest),
                                                                     reinterpret_cast<const dtype&>(desired),
                                                                     reinterpret_cast<const dtype&>(expected));
               
               return oldDestination == reinterpret_cast<const dtype&>(expected);
           }
           else if constexpr (sizeof(T) * 8 == 32)
           {
               // LONG -- 32-bit signed integer
               using dtype = LONG;
               assert(intptr_t(dest) % sizeof(dtype) == 0); // Ddest should be aligned by 4 bytes
               LONG oldDestination = ::InterlockedCompareExchange(reinterpret_cast<volatile dtype*>(dest),
                                                                  reinterpret_cast<const dtype&>(desired),
                                                                  reinterpret_cast<const dtype&>(expected));

               return oldDestination == reinterpret_cast<const dtype&>(expected);
           }
           else if constexpr (sizeof(T) * 8 == 64)
           {
               // LONG64 -- 64-bit signed integer.
               using dtype = LONG64;
               assert(intptr_t(dest) % sizeof(dtype) == 0); // Ddest should be aligned by 8 bytes

               dtype oldDestination = ::InterlockedCompareExchange64(reinterpret_cast<volatile dtype*>(dest),
                                                                      reinterpret_cast<const dtype&>(desired),
                                                                      reinterpret_cast<const dtype&>(expected));

               return oldDestination == reinterpret_cast<const dtype&>(expected);
           }
           else
           {
               assert(!"UNLIKELY CASE. BACKUP IMPL. PLEASE CHECK.");
               
               static burt::DefaultMutex m;
               m.lock();
               
               T oldDestination = *dest;
               if (oldDestination == expected)
               {
                   *dest = desired;
               }
               
               m.unlock();
               
               return oldDestination == expected;
           }
        }
                   
        /** Threading based on WinApi
        */
        class ThreadWinApi
        {
        public:
            typedef int32_t(*StartRoutine)(void* arg1, void* arg2);

            /** Create thread in current process
            * @param start start routine
            * @param userArg extra arg to to thread startup code
            * @param reserverStackSize stack size which need to be reserved for stack.
            */
            explicit ThreadWinApi(StartRoutine start, void* userArg1 = 0, void* userArg2 = 0, size_t reserverStackSize = 0);

            ~ThreadWinApi();

            ThreadWinApi(const ThreadWinApi&) = delete;

            ThreadWinApi(ThreadWinApi&&) = delete;

            /** Is thread alive (continue to execute)
            * @param returnCode is non zero the return code will be place here
            * @return true - if thread is still alive
            */
            bool isAlive(int32_t* returnCode = 0) const;

            /** Wait for end of the thread until milliseconds
            * @param milliseconds wait timeout
            */
            void join(uint32_t milliseconds);

            /** Wait until the thread will finished its work
            */
            void join();

            /** Get exit code of the terminated thread
            */
            int32_t getExitCode();

            /** Suspend thread execution
            * @sa resume
            */
            void suspend();

            /** Resume thread execution
            * @sa suspend
            */
            void resume();
            
            /** Sleep current thread until timeout ms
            * @param milliseconds timeout
            */
            static void sleepCurrentTh(uint32_t milliseconds);

            /** Nothing todo in current thread. Give execution to another thread.
            */
            static void yeildCurrentTh();

            /** Nothing todo in current thread. Optionally give execution to another thread.
            */
            static void yeildCurrentThInHotLoop();
            
            /** Setup hard affinity (or binding mask) for current executed CPU thread.
            * This call will allow execution of the thread only on specific processors
            * @param mask mask which contain bitwise "1" in corresponding position in which processors it's possible to execute thread
            */
            static void setThreadAffinityMaskForCurrentTh(uint64_t mask);

            /** Get hard affinity (or binding mask) for current executed CPU thread.
            * @return current affinity mask for specific thread
            */
            static uint64_t getThreadAffinityMaskForCurrentTh();

            /** Setup hard affinity (or binding mask) for current executed CPU thread.
            * This call will allow execution of the thread only on specific processors
            * @param mask mask which contain bitwise "1" in corresponding position in which processors it's possible to execute thread
            */
            void setThreadAffinityMask(uint64_t mask);

            /** Get hard affinity (or binding mask) for current executed CPU thread.
            * @return current affinity mask for specific thread
            */
            uint64_t getThreadAffinityMask();
            
        private:
            StartRoutine start;

            void* userArg1;
            void* userArg2;

            HANDLE threadHandle;
            DWORD threadID;

            friend DWORD WINAPI InternalWorkerThreadWinApi(LPVOID lpParam);
        };
    }
}

#endif
