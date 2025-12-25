#include <iostream>
#include <memory>
#include "autograd/value.hpp"
#include "autograd/ops.hpp"
#include "autograd/backward.hpp"
#include "autograd/constant.hpp"
#include "autograd/activations.hpp"

using namespace autograd;

int main() {

    std::cout << "=== Test 1: ReLU (no chain) ===\n";

    // Case 1: x > 0
    {
        auto x = constant(3.0);
        auto y = relu(x);

        backward(y);

        std::cout << "x = 3\n";
        std::cout << "y = relu(x) = " << y->value << " (expected 3)\n";
        std::cout << "x.grad = " << x->grad << " (expected 1)\n\n";
    }

    // Case 2: x < 0
    {
        auto x = constant(-2.0);
        auto y = relu(x);

        backward(y);

        std::cout << "x = -2\n";
        std::cout << "y = relu(x) = " << y->value << " (expected 0)\n";
        std::cout << "x.grad = " << x->grad << " (expected 0)\n\n";
    }
 std::cout << "=== Test 2: ReLU + chain rule ===\n";

    // Case 1: x > 0
    {
        auto x = constant(2.0);
        auto y = relu(x);
        auto z = mult(y, constant(5.0));  // z = relu(x) * 5

        backward(z);

        std::cout << "x = 2\n";
        std::cout << "z = relu(x) * 5 = " << z->value << " (expected 10)\n";
        std::cout << "x.grad = " << x->grad << " (expected 5)\n\n";
    }

    // Case 2: x < 0
    {
        auto x = constant(-2.0);
        auto y = relu(x);
        auto z = mult(y, constant(5.0));

        backward(z);

        std::cout << "x = -2\n";
        std::cout << "z = relu(x) * 5 = " << z->value << " (expected 0)\n";
        std::cout << "x.grad = " << x->grad << " (expected 0)\n\n";
    }

    return 0;
}