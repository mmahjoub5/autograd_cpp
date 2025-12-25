#include "autograd/value.hpp"
#include "autograd/graph_utils.hpp"
namespace autograd {
    void backward(std::shared_ptr<Value> loss);
}