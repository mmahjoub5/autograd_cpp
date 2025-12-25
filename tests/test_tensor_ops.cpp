#include <iostream>
#include <memory>

#include "autograd/value.hpp"
#include "autograd/tensor.hpp"
#include "autograd/ops.hpp"
#include "autograd/backward.hpp"

using autograd::Value;
using autograd::Tensor;
using autograd::dot;
using autograd::backward;

int main() {
    // ---- Tensor a: shape (1, 3) ----
    auto a = std::make_shared<Tensor>();
    a->shape = {1, 3};

    for (double v : {1.0, 2.0, 3.0}) {
        auto val = std::make_shared<Value>();
        val->value = v;
        val->grad = 0.0;
        a->values.push_back(val);
    }

    // ---- Tensor b: shape (3, 1) ----
    auto b = std::make_shared<Tensor>();
    b->shape = {3, 1};

    for (double v : {4.0, 5.0, 6.0}) {
        auto val = std::make_shared<Value>();
        val->value = v;
        val->grad = 0.0;
        b->values.push_back(val);
    }

    // ---- Dot product ----
    auto out = dot(a, b);

    std::cout << "Dot product value: " << out->value << std::endl;

    // ---- Backward pass ----
    backward(out);

    std::cout << "\nGradients wrt a:\n";
    for (size_t i = 0; i < a->values.size(); ++i) {
        std::cout << "da[" << i << "] = " << a->values[i]->grad << "\n";
    }

    std::cout << "\nGradients wrt b:\n";
    for (size_t i = 0; i < b->values.size(); ++i) {
        std::cout << "db[" << i << "] = " << b->values[i]->grad << "\n";
    }

    return 0;
}
