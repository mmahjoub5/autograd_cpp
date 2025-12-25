#include "autograd/graph_utils.hpp"

namespace autograd {
    void topSort(std::shared_ptr<Value> node, std::unordered_set<std::shared_ptr<Value>> &V, std::vector<std::shared_ptr<Value>> & topSortedNodes) {
        if (V.find(node) != V.end()) {
            return;
        }
        if (node->parents.empty() ) {
            V.insert(node);
            topSortedNodes.push_back(node);
            return;
        }
        for (auto parent : node->parents) {
            if (V.find(parent) == V.end()) {
                topSort(parent, V, topSortedNodes);
            }
        }
        V.insert(node);
        topSortedNodes.push_back(node);
    }
}