#pragma once 

#include "autograd/value.hpp"
namespace autograd {
    struct Tensor {
         // ===== Forward (primal) =====
        std::vector<std::shared_ptr<Value>> values;
        
        // ===== Shape =====
        std::vector<int> shape;
    };

    std::shared_ptr<Tensor> create_tensor(std::vector<float> data, int rows, int cols, bool requires_grad=true);
    std::vector<std::shared_ptr<Value>> create_matrix(std::vector<float> data, int rows, int cols, bool requires_grad=true);
    std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> A);
}