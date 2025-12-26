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
    auto bias = create_tensor({0.5, -1.0}, 2, 1);          // 2x1 bias vector
    auto losses = std::vector<float>{};
    // Forward pass: compute weighted sum
    for (int i = 0; i < 200 ; i++ ) {
        auto weights_t = transpose(weights); // Transpose weights to 2x2
        //print transposed shape
        std::cout << "Weights transposed shape: (" << weights_t->shape[0] << ", " << weights_t->shape[1] << ")\n";
        auto matmul_out = matmul(weights_t, inputs); // 2x1 output vector
        std::cout << "Matmul output: ";
        for (const auto& val : matmul_out->values) {
            std::cout << val->value << " ";
        }
        std::cout << "matmul size" << matmul_out->shape[0] << " " << matmul_out->shape[1] << std::endl;


        auto output = addBias(matmul_out, bias); // 2x1 output vector

        std::cout << "Output of bias: ";
        for (const auto& val : output->values) {
            std::cout << val->value << " ";
        }
        std::cout << std::endl;
        auto relu1 = relu(output->values[0]);
        auto relu2 = relu(output->values[1]);
        // Combine relu outputs into a single loss value (sum)

        auto loss = add(relu1, relu2);
        std::cout << "Loss: " << loss->value << std::endl;
        losses.push_back(loss->value);
        // Backward pass: compute gradients
        backward(loss);
        std::cout << "Gradients wrt weights:\n";

        for (size_t i = 0; i < weights->values.size(); ++i) {
            std::cout << "dw[" << i << "] = " << weights->values[i]->grad << "\n";
            //std::cout << "dw[" << i << "] = " << inputs->values[i]->grad << "\n";
            std::cout << "---------------------\n";
        }

        //print bias gradients
        std::cout << "Gradients wrt bias:\n";
        for (size_t i = 0; i < bias->values.size(); ++i) {
            std::cout << "db[" << i << "] = " << bias->values[i]->grad << "\n";
            std::cout << "---------------------\n"; 
        }


        // update weights with a simple SGD step
        double learning_rate = 0.01;
        for (size_t i = 0; i < weights->values.size(); ++i)
        {
            weights->values[i]->value -= learning_rate * weights->values[i]->grad;
        }
        for (size_t i = 0; i < bias->values.size(); ++i)
        {
            bias->values[i]->value -= learning_rate * bias->values[i]->grad;
        }
        
        // Reset gradients for next iteration
        for (size_t i = 0; i < weights->values.size(); ++i)
        {
            weights->values[i]->grad = 0.0; 
        }
        for (size_t i = 0; i < bias->values.size(); ++i)
        {
            bias->values[i]->grad = 0.0;
        }   
    }
    // Print all losses
    std::cout << "Losses over iterations: ";
    for (const auto& l : losses) {
        std::cout << l << " ";
    }
    std::cout << std::endl;

    // Final weights
    std::cout << "Final weights: ";
    for (const auto& w : weights->values) {
        std::cout << w->value << " ";
    }
    std::cout << std::endl;
    

    return 0;   
}