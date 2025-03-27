#include <torch/torch.h>
#include <iostream>
#include <chrono>  // For timing
#include <vector>  // For storing tensors

int main() 
{
    // Print PyTorch version
    std::cout << "Torch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << std::endl;


    // Output the last values of g, a.grad, and b.grad
    torch::Tensor a = torch::tensor(-4.0, torch::dtype(torch::kFloat64).requires_grad(true));
    torch::Tensor b = torch::tensor(2.0, torch::dtype(torch::kFloat64).requires_grad(true));

    torch::Tensor c = a + b;
    torch::Tensor d = a * b + torch::pow(b, 3);

    c = c + c + 1;
    c = c + 1 + c + (-a);

    d = d + d * 2 + (torch::relu(b + a));
    d = d + 3 * d + (torch::relu(b - a));

    torch::Tensor e = c - d;
    torch::Tensor f = torch::pow(e, 2);
    torch::Tensor g = f / 2.0;
    g = g + 10.0 / f;
    g.backward();

    std::cout << "g: " << g.item<double>() << std::endl;
    std::cout << "a.grad: " << a.grad().item<double>() << std::endl;
    std::cout << "b.grad: " << b.grad().item<double>() << std::endl;


    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    // Save all tensors to a file
    for (size_t i = 0; i < 5000; ++i)
    {
        std::vector<torch::Tensor> tensors = { a, b, c, d, e, f, g };
        torch::save(tensors, "saved_tensors_libtorch.bin");
    }


    // End the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print results
    std::cout << "Time: " << elapsed.count() << " seconds\n";

    //getchar();
    return 0;
}
