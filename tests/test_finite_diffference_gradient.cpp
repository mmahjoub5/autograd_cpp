#include <iostream>
#include <memory>
#include "autograd/value.hpp"
#include "autograd/ops.hpp"
#include "autograd/backward.hpp"
#include "autograd/constant.hpp"
#include "autograd/activations.hpp"
#include "autograd/tensor.hpp"
using namespace autograd;

double forward_loss_only(
    std::shared_ptr<Tensor> weights,
    std::shared_ptr<Tensor> inputs,
    std::shared_ptr<Tensor> bias
) {
    auto weights_t  = transpose(weights);
    auto matmul_out = matmul(weights_t, inputs);   // (2,1)
    auto output     = addBias(matmul_out, bias);   // (2,1)

    auto r0 = relu(output->values[0]);
    auto r1 = relu(output->values[1]);
    auto loss = add(r0, r1);

    return loss->value;
}

void finite_difference_check(
    std::shared_ptr<Tensor> weights,
    std::shared_ptr<Tensor> inputs,
    std::shared_ptr<Tensor> bias,
    int idx
) {
    double eps = 1e-4;

    // Save original value
    double original = weights->values[idx]->value;

    // f(w + eps)
    weights->values[idx]->value = original + eps;
    double L_plus = forward_loss_only(weights, inputs, bias);

    // f(w - eps)
    weights->values[idx]->value = original - eps;
    double L_minus = forward_loss_only(weights, inputs, bias);

    // Restore
    weights->values[idx]->value = original;

    double numerical_grad = (L_plus - L_minus) / (2.0 * eps);
    double autograd_grad  = weights->values[idx]->grad;

    std::cout << "Gradient check for w[" << idx << "]\n";
    std::cout << "  autograd  = " << autograd_grad << "\n";
    std::cout << "  numerical = " << numerical_grad << "\n";
    std::cout << "  abs diff  = " << std::abs(autograd_grad - numerical_grad) << "\n";
}



int main() {
    // --- Build the same toy graph as before ---
    auto weights = create_tensor({0.2, 0.8, -0.5, 1.0}, 2, 2);      // (2,2)
    auto inputs  = create_tensor({1.0, 2.0}, 2, 1, false);          // (2,1) not trainable
    auto bias    = create_tensor({0.5, -1.0}, 2, 1);                // (2,1)

    // Forward (with graph)
    auto weights_t  = transpose(weights);
    std::cout << "Weights transposed shape: ("
              << weights_t->shape[0] << ", " << weights_t->shape[1] << ")\n";

    auto matmul_out = matmul(weights_t, inputs);                    // (2,1)
    std::cout << "Matmul output: ";
    for (const auto& v : matmul_out->values) std::cout << v->value << " ";
    std::cout << "matmul size " << matmul_out->shape[0] << " " << matmul_out->shape[1] << "\n";

    auto output = addBias(matmul_out, bias);                        // (2,1)
    std::cout << "Output of bias: ";
    for (const auto& v : output->values) std::cout << v->value << " ";
    std::cout << "\n";

    auto r0 = relu(output->values[0]);
    auto r1 = relu(output->values[1]);
    auto loss = add(r0, r1);

    std::cout << "Loss: " << loss->value << "\n";

    // Backward (autograd grads)
    backward(loss);

    std::cout << "Gradients wrt weights:\n";
    for (size_t i = 0; i < weights->values.size(); ++i) {
        std::cout << "dw[" << i << "] = " << weights->values[i]->grad << "\n";
        std::cout << "---------------------\n";
    }

    std::cout << "Gradients wrt bias:\n";
    for (size_t i = 0; i < bias->values.size(); ++i) {
        std::cout << "db[" << i << "] = " << bias->values[i]->grad << "\n";
        std::cout << "---------------------\n";
    }

    // Finite-difference checks (pick a couple indices; these two should be non-zero in your setup)
    finite_difference_check(weights, inputs, bias, 1);
    finite_difference_check(weights, inputs, bias, 3);

    return 0;
}
