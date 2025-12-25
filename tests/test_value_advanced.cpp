#include <iostream>
#include <memory>

#include "autograd/value.hpp"
#include "autograd/ops.hpp"
#include "autograd/backward.hpp"

using autograd::Value;
using autograd::add;
using autograd::div;
using autograd::exp;
using autograd::backward;

int main() {
    std::cout << "=== Test: Exponential ===\n";
    {
        auto x = std::make_shared<Value>();
        x->value = 2.0;

        auto z = exp(x);
        backward(z);

        std::cout << "z.grad = " << z->grad << " (expected 1)\n";
        std::cout << "x.grad = " << x->grad << " (expected e^2)\n\n";
    }

    std::cout << "=== Test: Softmax-like computation ===\n";
    {
        auto x0 = std::make_shared<Value>(); x0->value = 1.0;
        auto x1 = std::make_shared<Value>(); x1->value = 2.0;

        auto e0 = exp(x0);
        auto e1 = exp(x1);

        auto sum = add(e0, e1);

        auto s0 = div(e0, sum);
        auto s1 = div(e1, sum);

        auto loss = add(s0, s1);

        backward(loss);

        std::cout << "s0 = " << s0->value << "\n";
        std::cout << "s1 = " << s1->value << "\n";

        std::cout << "x0.grad = " << x0->grad << " (expected ~0)\n";
        std::cout << "x1.grad = " << x1->grad << " (expected ~0)\n";
    }

    return 0;
}
