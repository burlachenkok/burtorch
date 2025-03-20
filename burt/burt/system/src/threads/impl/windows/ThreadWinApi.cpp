#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/system/include/threads/impl/windows/ThreadWinApi.h"

#include <iostream>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if BURT_WINDOWS

#include <process.h> 

namespace burt
{
    namespace internal
    {
        DWORD WINAPI InternalWorkerThreadWinApi(LPVOID lpParam)
        {            
            ThreadWinApi* thContext = static_cast<ThreadWinApi*>(lpParam);
            return thContext->start(thContext->userArg1, thContext->userArg2);
        }

        ThreadWinApi::ThreadWinApi(ThreadWinApi::StartRoutine theStart, void* theUserArg1, void* theUserArg2, size_t reserverStackSize)
            : start(theStart)
            , userArg1(theUserArg1)
            , userArg2(theUserArg2)
            , threadHandle(0)
            , threadID(0)
        {
            typedef unsigned (__stdcall* PTHREAD_START) (void*);
            
            constexpr bool kInititializeWithCRuntime = true;
            
            if (kInititializeWithCRuntime)
            {
                // Create Thread with initialized of C++/C runtime
                threadHandle = (HANDLE) _beginthreadex((void*)NULL,
                                                        reserverStackSize, 
                                                        (PTHREAD_START) InternalWorkerThreadWinApi,
                                                        (void*) this, 
                                                        (unsigned int) CREATE_SUSPENDED, 
                                                        (unsigned*) &threadID );

            }
            else
            {
                // Create Thread without initialized of C++/C runtime (but which by Microsoft promise will be initialized once thread will touch CRT global state)
                threadHandle = ::CreateThread(NULL, 
                                              reserverStackSize, 
                                              InternalWorkerThreadWinApi, 
                                              this, 
                                              CREATE_SUSPENDED, 
                                              &threadID);
            }
            
            if (threadHandle == NULL)
            {
                std::cerr << "WinAPI: Problems with creating one more thread.\n";
                abort();
            }
            else
            {
                // Start thread
                ResumeThread(threadHandle);
            }
        }

        ThreadWinApi::~ThreadWinApi()
        {
            if (isAlive())
            {
                std::cerr << "WARNING: You're trying to call destructor for not completed thread!\n";
            }
            
            ::CloseHandle(threadHandle);
        }

        bool ThreadWinApi::isAlive(int32_t* returnCode) const
        {
            if (WaitForSingleObject(threadHandle, 0) == WAIT_TIMEOUT)
                return true;

            if (returnCode)
            {
                DWORD exitCode;
                GetExitCodeThread(threadHandle, &exitCode);
                *returnCode = exitCode;
            }
            return false;
        }

        void ThreadWinApi::join(uint32_t milliseconds)
        {
            ::WaitForSingleObject(threadHandle, milliseconds);
        }

        void ThreadWinApi::join()
        {
            ::WaitForSingleObject(threadHandle, INFINITE);
        }

        int32_t ThreadWinApi::getExitCode()
        {
            DWORD exitCode;
            GetExitCodeThread(threadHandle, &exitCode);
            int32_t exitCodei32 = exitCode;

            return exitCodei32;
        }

        void ThreadWinApi::suspend()
        {
            ::SuspendThread(threadHandle);
        }

        void ThreadWinApi::resume()
        {
            ::ResumeThread(threadHandle);
        }

        void ThreadWinApi::sleepCurrentTh(uint32_t milliseconds)
        {
            ::Sleep(milliseconds);
        }

        void ThreadWinApi::yeildCurrentTh()
        {
            ::SwitchToThread();
        }
        
        void ThreadWinApi::yeildCurrentThInHotLoop()
        {
#if 0
            yeildCurrentTh();
#else
            constexpr uint16_t kSpinCounter = 2000;

            static thread_local uint16_t spinCounter4CurrentThread = 0;
            spinCounter4CurrentThread++;

            if (spinCounter4CurrentThread == kSpinCounter)
            {
                spinCounter4CurrentThread = 0;
                yeildCurrentTh();
            }
#endif
        }

        void ThreadWinApi::setThreadAffinityMaskForCurrentTh(uint64_t mask)
        {
            SetThreadAffinityMask(GetCurrentThread(), mask);
        }

        uint64_t ThreadWinApi::getThreadAffinityMaskForCurrentTh()
        {
            DWORD_PTR mask = SetThreadAffinityMask(GetCurrentThread(), 0x1);
            SetThreadAffinityMask(GetCurrentThread(), mask);
            return mask;
        }

        void ThreadWinApi::setThreadAffinityMask(uint64_t mask)
        {
            SetThreadAffinityMask(threadHandle, mask);
        }

        uint64_t ThreadWinApi::getThreadAffinityMask()
        {
            DWORD_PTR mask = SetThreadAffinityMask(threadHandle, 0x1);
            SetThreadAffinityMask(threadHandle, mask);
            return mask;
        }
    }
}

#endif
