#include <torch/torch.h>
#include <iostream>
#include <chrono>

int main() 
{
    std::cout << "Torch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100 * 1000; ++i) {
        // Initialize tensors with requires_grad=true
        torch::Tensor a = torch::tensor(-41.0, torch::requires_grad().dtype(torch::kFloat64));
        torch::Tensor b = torch::tensor(2.0, torch::requires_grad().dtype(torch::kFloat64));

        // Perform computations
        torch::Tensor c = a + b;
        torch::Tensor d = a * b + torch::pow(b, 3);
        torch::Tensor e = c - d;
        torch::Tensor f = torch::pow(e, 2);
        torch::Tensor g = f / 2.0;

        // Perform backpropagation
        g.backward();
    }

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    // Print results
    std::cout << "Time: " << duration << " seconds" << std::endl;

    torch::Tensor a = torch::tensor(-41.0, torch::requires_grad().dtype(torch::kFloat64));
    torch::Tensor b = torch::tensor(2.0, torch::requires_grad().dtype(torch::kFloat64));

    torch::Tensor c = a + b;
    torch::Tensor d = a * b + torch::pow(b, 3);
    torch::Tensor e = c - d;
    torch::Tensor f = torch::pow(e, 2);
    torch::Tensor g = f / 2.0;

    g.backward();

    std::cout << "g: " << g.item<double>() << std::endl;
    std::cout << "a.grad: " << a.grad().item<double>() << std::endl;
    std::cout << "b.grad: " << b.grad().item<double>() << std::endl;

    return 0;
}
