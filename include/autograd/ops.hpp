#pragma once
#include "autograd/value.hpp"
#include "autograd/tensor.hpp"

namespace autograd {
    std::shared_ptr<Value> add( std::shared_ptr<Value> x, std::shared_ptr<Value> y);
    std::shared_ptr<Value> mult( std::shared_ptr<Value> x, std::shared_ptr<Value> y);
    std::shared_ptr<Value> sub( std::shared_ptr<Value> x, std::shared_ptr<Value> y);
    std::shared_ptr<Value> div( std::shared_ptr<Value> x, std::shared_ptr<Value> y);
    std::shared_ptr<Value> exp( std::shared_ptr<Value> x);
    std::shared_ptr<Value> log( std::shared_ptr<Value> x);
    std::shared_ptr<Value> dot(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Value> max(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
}
