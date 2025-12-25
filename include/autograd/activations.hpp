#include <autograd/value.hpp>
#include  "autograd/ops.hpp"
#include "autograd/constant.hpp"

namespace autograd {
    std::shared_ptr<Value> relu(std::shared_ptr<Value> x); 
}