#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <iostream>
\
namespace autograd {

    struct Value {
        Value() noexcept;   // ← THIS LINE IS REQUIRED
        // ===== Forward (primal) =====
        double value;  

        // ===== Backward (adjoint) =====
        double grad;

        // ===== Graph structure =====
        std::vector<std::shared_ptr<Value>> parents;

        // ===== Local backward rule =====
        std::function<void()> grad_fn;

        bool requires_grad = true;
    };

} // namespace autograd
