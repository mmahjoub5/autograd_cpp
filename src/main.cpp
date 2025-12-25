#include <iostream>

int main() {
    std::cout << "Autograd C++ Library\n";
    std::cout << "=====================\n\n";
    std::cout << "Test files have been moved to the tests/ directory.\n";
    std::cout << "To run the tests, build the project and execute:\n";
    std::cout << "  ./build/test_value_basic\n";
    std::cout << "  ./build/test_value_arithmetic\n";
    std::cout << "  ./build/test_value_advanced\n";
    std::cout << "  ./build/test_tensor_ops\n";
    std::cout << "\nOr run all tests with:\n";
    std::cout << "  cd build && ./test_value_basic && ./test_value_arithmetic && ./test_value_advanced && ./test_tensor_ops\n";
    
    return 0;
}
