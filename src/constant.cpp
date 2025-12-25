#include "autograd/constant.hpp"

namespace autograd {
    std::shared_ptr<Value> constant(double v) {
        auto out = std::make_shared<Value>();
        out->value = v;
        out->grad = 0.0;
        // No parents since it's a constant
        out->grad_fn = nullptr; // No gradient function for constants
        return out;
    }
}