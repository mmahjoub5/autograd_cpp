#include <iostream>
#include <memory>
#include "autograd/value.hpp"
#include "autograd/ops.hpp"
#include "autograd/backward.hpp"
#include "autograd/constant.hpp"
#include "autograd/activations.hpp"
#include "autograd/tensor.hpp"
using namespace autograd;

int main() {

    auto weights = create_tensor({0.2, 0.8, -0.5, 1.0}, 2, 2); // 2x2 weight matrix
    auto inputs = create_tensor({1.0, 2.0}, 2, 1, false);          // 2x1 input vector
    
    // Forward pass: compute weighted sum

    auto weights_t = transpose(weights); // Transpose weights to 2x2
    //print transposed shape
    std::cout << "Weights transposed shape: (" << weights_t->shape[0] << ", " << weights_t->shape[1] << ")\n";

    auto output = matmul(weights, inputs); // 2x1 output vector

    std::cout << "Output: ";
    for (const auto& val : output->values) {
        std::cout << val->value << " ";
    }
    std::cout << std::endl;
    auto relu1 = relu(output->values[0]);
    auto relu2 = relu(output->values[1]);
    // Combine relu outputs into a single loss value (sum)

    auto loss = add(relu1, relu2);
    std::cout << "Loss: " << loss->value << std::endl;
    // Backward pass: compute gradients
    backward(loss);
    std::cout << "Gradients wrt weights:\n";

    for (size_t i = 0; i < weights->values.size(); ++i) {
        std::cout << "dw[" << i << "] = " << weights->values[i]->grad << "\n";
        //std::cout << "dw[" << i << "] = " << inputs->values[i]->grad << "\n";
    }
    
    return 0;   
}