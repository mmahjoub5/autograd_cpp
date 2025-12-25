#include <iostream>
#include <memory>

#include "autograd/value.hpp"
#include "autograd/ops.hpp"
#include "autograd/backward.hpp"

using autograd::Value;
using autograd::add;
using autograd::backward;

int main() {
    std::cout << "=== Test 1: Leaf ===\n";
    {
        auto x = std::make_shared<Value>();
        x->value = 3.0;

        backward(x);

        std::cout << "x.grad = " << x->grad << " (expected 1)\n\n";
    }

    std::cout << "=== Test 2: Addition ===\n";
    {
        auto x = std::make_shared<Value>();
        auto y = std::make_shared<Value>();
        x->value = 2.0;
        y->value = 3.0;

        auto z = add(x, y);

        backward(z);

        std::cout << "z.grad = " << z->grad << " (expected 1)\n";
        std::cout << "x.grad = " << x->grad << " (expected 1)\n";
        std::cout << "y.grad = " << y->grad << " (expected 1)\n\n";
    }

    std::cout << "=== Test 3: Chain rule ===\n";
    {
        auto x = std::make_shared<Value>();
        auto y = std::make_shared<Value>();
        x->value = 2.0;
        y->value = 3.0;

        auto z = add(x, y);
        auto w = add(z, z);

        backward(w);

        std::cout << "w.grad = " << w->grad << " (expected 1)\n";
        std::cout << "z.grad = " << z->grad << " (expected 2)\n";
        std::cout << "x.grad = " << x->grad << " (expected 2)\n";
        std::cout << "y.grad = " << y->grad << " (expected 2)\n";
    }

    return 0;
}
