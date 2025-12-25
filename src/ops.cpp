#include "autograd/ops.hpp"

namespace autograd {
    std::shared_ptr<Value> add(std::shared_ptr<Value> x, std::shared_ptr<Value> y) {
        auto out = std::make_shared<Value>();
        out->parents.push_back(x);
        out->parents.push_back(y);
        out->value = x->value + y->value;

        out->grad_fn = [x, y, out]() {
            x->grad += out->grad;
            y->grad += out->grad;
        };
        return out;
    }
    std::shared_ptr<Value> mult(std::shared_ptr<Value> x, std::shared_ptr<Value> y) {
        auto out = std::make_shared<Value>();
        out->parents.push_back(x);
        out->parents.push_back(y);
        out->value = x->value * y->value;

        out->grad_fn = [x, y, out]() {
            x->grad += out->grad * y->value;
            y->grad += out->grad * x->value;
        };
        return out;
    }
     std::shared_ptr<Value> sub( std::shared_ptr<Value> x, std::shared_ptr<Value> y) {
        auto out = std::make_shared<Value>();
        out->parents.push_back(x);
        out->parents.push_back(y);
        out->value = x->value - y->value;

        out->grad_fn = [x, y, out]() {
            x->grad += out->grad;
            y->grad -= out->grad;
        };
        return out;
     }
     std::shared_ptr<Value> div( std::shared_ptr<Value> x, std::shared_ptr<Value> y) {
        auto out = std::make_shared<Value>();
        out->parents.push_back(x);
        out->parents.push_back(y);
        out->value = x->value / y->value;

        out->grad_fn = [x, y, out]() {
            x->grad += out->grad / y->value;
            y->grad -= out->grad * x->value / (y->value * y->value);
        };
        return out;
     }

     std::shared_ptr<Value> exp( std::shared_ptr<Value> x) {
        auto out = std::make_shared<Value>();
        out->parents.push_back(x);
        out->value = std::exp(x->value);

        out->grad_fn = [x, out]() {
            x->grad += out->grad * out->value;
        };
        return out;
     }
    std::shared_ptr<Value> log( std::shared_ptr<Value> x) {
        auto out = std::make_shared<Value>();
        out->parents.push_back(x);
        out->value = std::log(x->value);

        out->grad_fn = [x, out]() {
            x->grad += out->grad / x->value;
        };
        return out;
    }
    std::shared_ptr<Value> dot(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        // check dimensions 
        if (a->shape[1] != b->shape[0])     {
            throw std::invalid_argument("Incompatible tensor shapes for dot product");
        }
        
        auto out = std::make_shared<Value>();
        out= mult(a->values[0], b->values[0]);

        for (int i = 1; i < a->shape[1]; ++i) {
            out = add(out, mult(a->values[i], b->values[i]));
        }

        return out;
        
    }


}