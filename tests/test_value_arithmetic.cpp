#include <iostream>
#include <memory>

#include "autograd/value.hpp"
#include "autograd/ops.hpp"
#include "autograd/backward.hpp"

using autograd::Value;
using autograd::mult;
using autograd::div;
using autograd::backward;

int main() {
    std::cout << "=== Test: Multiplication ===\n";
    {
        auto x = std::make_shared<Value>();
        auto y = std::make_shared<Value>();
        x->value = 2.0;
        y->value = 3.0;

        auto z = mult(x, y);
        backward(z);

        std::cout << "z.grad = " << z->grad << " (expected 1)\n";
        std::cout << "x.grad = " << x->grad << " (expected 3)\n";
        std::cout << "y.grad = " << y->grad << " (expected 2)\n\n";
    }

    std::cout << "=== Test: Chain multiplication ===\n";
    {
        auto x = std::make_shared<Value>();
        auto y = std::make_shared<Value>();
        x->value = 2.0;
        y->value = 3.0;

        auto z = mult(x, y);
        auto w = mult(z, y);
        backward(w);

        std::cout << "w.grad = " << w->grad << " (expected 1)\n";
        std::cout << "z.grad = " << z->grad << " (expected 3)\n";
        std::cout << "x.grad = " << x->grad << " (expected 9)\n";
        std::cout << "y.grad = " << y->grad << " (expected 12)\n\n";
    }

    std::cout << "=== Test: Division ===\n";
    {
        auto x = std::make_shared<Value>();
        auto y = std::make_shared<Value>();
        x->value = 10.0;
        y->value = 2.0;

        auto z = div(x, y);
        backward(z);

        std::cout << "z.grad = " << z->grad << " (expected 1)\n";
        std::cout << "x.grad = " << x->grad << " (expected 0.5)\n";
        std::cout << "y.grad = " << y->grad << " (expected -2.5)\n\n";
    }

    std::cout << "=== Test: Chain division ===\n";
    {
        auto x = std::make_shared<Value>();
        auto y = std::make_shared<Value>();
        x->value = 10.0;
        y->value = 2.0;

        auto z = div(x, y);
        auto w = div(z, y);
        backward(w);

        std::cout << "w.grad = " << w->grad << " (expected 1)\n";
        std::cout << "z.grad = " << z->grad << " (expected 0.5)\n";
        std::cout << "x.grad = " << x->grad << " (expected 0.25)\n";
        std::cout << "y.grad = " << y->grad << " (expected -2.5)\n\n";
    }

    return 0;
}
