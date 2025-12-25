#include "autograd/value.hpp"

namespace autograd {

    Value::Value() noexcept : value(0.0), grad(0.0), parents(), grad_fn(nullptr) {}
    
}