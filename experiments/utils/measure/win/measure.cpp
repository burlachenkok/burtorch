// Small program that executes an external command provided as an argument, measures its execution time, and prints the duration in milliseconds.
// OS: Windows

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <process.h>

int main(int argc, char** argv) 
{
    if (argc < 2) 
    {
        fprintf(stderr, "** Error: no command provided\n");
        exit(1);
    }

    printf("Running ");
    for (int i = 1; i < argc; ++i)
    {
        printf("[%s] ", argv[i]);
    }
    printf("\n");

    auto handle = _spawnvp(_P_NOWAIT, argv[1], &argv[1]);
    if (handle < 0) 
    {
        perror("_spawnv failed");
        exit(1);
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    int status;
    if (_cwait(&status, handle, _WAIT_CHILD) == -1) 
    {
        perror("_cwait failed");
        exit(1);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    printf("duration: %lld ms\n", (long long int)duration.count());
}
