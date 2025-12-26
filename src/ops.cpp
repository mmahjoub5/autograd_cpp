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

    std::shared_ptr<Value> max(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
        auto out = std::make_shared<Value>();
        out->parents.push_back(a);
        out->parents.push_back(b);

        if (a->value >= b->value) {
            out->value = a->value;
            out->grad_fn = [a,b, out]() {
                a->grad += out->grad;
                b->grad += 0.0;
            };
        } 
        else {
            out->value = b->value;
            out->grad_fn = [a, b, out]() {
                b->grad += out->grad;
                a->grad += 0.0;
            };

        }
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

    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        // check dimensions 
        if (a->shape[1] != b->shape[0])     {
            throw std::invalid_argument("Incompatible tensor shapes for matrix multiplication");
        }
        // index for a flat vector index = i * col + j
        auto out = std::make_shared<Tensor>();
        out->shape = {a->shape[0], b->shape[1]};
        auto out_values = std::vector<std::shared_ptr<Value>>();
        auto inner = a->shape[1];
       
        for (int i = 0; i < a->shape[0]; ++i) {
            for (int j = 0; j < b->shape[1]; ++j) {
                std::vector<std::shared_ptr<Value>> a_row;
                std::vector<std::shared_ptr<Value>> b_col;
                for (int k = 0; k < inner; ++k) {
                    a_row.push_back(a->values[i * a->shape[1] + k]);
                    b_col.push_back(b->values[k * b->shape[1] + j]);
                }
                 auto dot_product = dot( std::make_shared<Tensor>(Tensor{a_row, {1, (int)a_row.size()}}), 
                            std::make_shared<Tensor>(Tensor{b_col, {(int)b_col.size(), 1}}) );
                out_values.push_back(dot_product);   
            }
           
        }
         
        out->values = out_values;
        return out;
    }

        std::shared_ptr<Tensor> addBias(std::shared_ptr<Tensor> X, std::shared_ptr<Tensor> b) {
            // Check dimensions
            if (X->shape[1] != b->shape[1] && X->shape[0] != b->shape[0]) {
                std::cout << X->shape[1] << " !=  " << b->shape[1] << std::endl;
                std::cout << X->shape[0] << " != " << b->shape[0] << std::endl;
                throw std::invalid_argument("Incompatible tensor shapes for addBias");
            }
            auto out = std::make_shared<Tensor>();
            out->shape = X->shape;
            std::vector<std::shared_ptr<Value>> out_values;
            for (int i = 0 ; i < X->shape[0]; ++i) {
                for (int j = 0; j < X->shape[1]; ++j) {
                    auto sum = add(X->values[i * X->shape[1] + j], b->values[i]);
                    out_values.push_back(sum);
                }
            }
            out->values = out_values;
            return out;

        }


}