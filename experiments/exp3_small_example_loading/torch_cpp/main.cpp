#include <torch/torch.h>
#include <iostream>
#include <chrono>  // For timing
#include <vector>  // For storing tensors

int main() 
{
    // Print PyTorch version
    std::cout << "Torch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << std::endl;

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    // Loop to load the tensors from the file 5000 times
    for (size_t i = 0; i < 5000; ++i)
    {
        std::vector<torch::Tensor> tensors;
        torch::load(tensors, "saved_tensors_libtorch.bin");
    }

    // End the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print results
    std::cout << "Time to load tensors 5000 times: " << elapsed.count() << " seconds\n";

    return 0;
}
