#include "autograd/value.hpp"
#include <unordered_set>
#include <vector>

namespace autograd {
    void topSort(std::shared_ptr<Value> node, std::unordered_set<std::shared_ptr<Value>> &V, std::vector<std::shared_ptr<Value>> & topSortedNodes);
}