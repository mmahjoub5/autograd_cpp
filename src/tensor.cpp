#include "autograd/tensor.hpp"

namespace autograd {
    // Tensor uses default constructor - no implementation needed
    std::vector<std::shared_ptr<Value>> create_matrix(std::vector<float> data, int rows, int cols, bool requires_grad) {
        std::vector<std::shared_ptr<Value>> matrix;
        for (int i = 0; i < rows * cols; ++i) {
            auto val = std::make_shared<Value>();
            val->value = data[i];
            val->grad = 0.0;
            val->requires_grad = true;
            matrix.push_back(val);
        }
        return matrix;
    }

    std::shared_ptr<Tensor> create_tensor(std::vector<float> data, int rows, int cols, bool requires_grad) {
        auto tensor = std::make_shared<Tensor>();
        tensor->shape = {rows, cols};
        tensor->values = create_matrix(data, rows, cols);
        return tensor;
    }

    std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> A) {
        int rows = A->shape[0];
        int cols = A->shape[1];
        auto transposed = std::make_shared<Tensor>();
        transposed->shape = {cols, rows};
        transposed->values = std::vector<std::shared_ptr<Value>>(rows * cols);
        for (int i = 0; i < A->shape[0]; ++i) {
            for (int j = 0; j < A->shape[1]; ++j) {
                transposed->values[j * A->shape[0] + i] = A->values[i * A->shape[1] + j] ;
            }
        }
        return transposed;
    }
}
