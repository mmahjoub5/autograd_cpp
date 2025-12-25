#include "autograd/backward.hpp"

namespace autograd {
    void backward(std::shared_ptr<Value> loss) {
        loss->grad = 1.0;
        auto vistedNodes = std::unordered_set<std::shared_ptr<Value>>();
        auto topoOrder = std::vector<std::shared_ptr<Value>>();
        topSort(loss, vistedNodes, topoOrder);
        for (auto node: topoOrder) {
            std::cout << "top sort output: ";
            std::cout << node->value << " \n";
        }
        for (auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it) {
            auto node = *it;
            if (node->grad_fn != nullptr) {
                node->grad_fn();
            }  
        }
    }
}
