#pragma once 

#include "autograd/value.hpp"
namespace autograd {
    struct Tensor {
         // ===== Forward (primal) =====
        std::vector<std::shared_ptr<Value>> values;
        
        // ===== Shape =====
        std::vector<int> shape;
    };
       
}