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
    std::cout << "=== Test: Add Bias to Tensor ===\n";
    auto weights = create_tensor({1, 1, 2, 2}, 2, 2); // 2x2 weight matrix
    auto bias = create_tensor({2, 2}, 2, 1);          // 2x1 input vector

    auto bias_out = addBias(weights, bias);
    
    // Forward pass: compute weighted sum
    // print bias  out values
    std::cout << "Bias added output: ";
    for (const auto& val : bias_out->values) {
        std::cout << val->value << " ";
    }
    std::cout << std::endl;

    
    return 0;   
}