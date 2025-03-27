#include <torch/torch.h>
#include <iostream>
#include <chrono>  // For timing

int main() {
    // Print PyTorch version
    std::cout << "Torch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << std::endl;

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    //std::cout << start.;

    for (int i = 0; i < 200000; ++i) {
        // Initialize tensors with requires_grad = true
        torch::Tensor a = torch::tensor(-4.0, torch::dtype(torch::kFloat64).requires_grad(true));
        torch::Tensor b = torch::tensor(2.0, torch::dtype(torch::kFloat64).requires_grad(true));

        // Perform the calculations
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

        // Perform backward pass
        g.backward();
    }

    // End the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print results
    std::cout << "Time: " << elapsed.count() << " seconds\n";

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

    //getchar();
    return 0;
}
