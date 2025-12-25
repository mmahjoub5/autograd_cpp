#include "autograd/activations.hpp"


namespace autograd {
    std::shared_ptr<Value> relu(std::shared_ptr<Value> x) {
        auto zero_value = constant(0.0);
        auto out = max(x, zero_value);
        return out;
    }
}