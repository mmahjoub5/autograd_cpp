# Autograd C++

A minimal **reverse-mode automatic differentiation** engine implemented from scratch in C++. This project provides a PyTorch-like autograd system for building and training neural networks with automatic gradient computation.

## Features

- **Scalar Automatic Differentiation** - Compute gradients automatically using computational graphs
- **Tensor Operations** - Matrix multiplication, element-wise operations, and more
- **Reverse-Mode Backpropagation** - Efficient gradient computation via topological sorting
- **Activation Functions** - ReLU and other non-linear functions
- **Comprehensive Test Suite** - Extensive tests including finite difference gradient checking
- **Memory Safe** - Built with smart pointers and modern C++ practices

## Project Structure

```
autograd_cpp/
├── include/autograd/      # Public API
├── src/                   # Implementation 
├── tests/                 # Test suite
├── build/                 # Build artifacts 
├── CMakeLists.txt         # CMake build 
├── run_test.sh            # Test execution
├── LICENSE                # Project license
└── README.md              # This file
```

## Algorithms & Techniques

### 1. **Reverse-Mode Automatic Differentiation**
The core algorithm implements reverse-mode autodiff (backpropagation) which efficiently computes gradients for functions with many inputs and few outputs (ideal for neural networks).

**Key Concept**: Build a computational graph during the forward pass, then traverse it backwards to accumulate gradients.

```cpp
// Forward pass builds the graph
auto a = std::make_shared<Value>(2.0);
auto b = std::make_shared<Value>(3.0);
auto c = add(a, b);    // c = a + b = 5.0

// Backward pass computes gradients
backward(c);     // dc/da = 1.0, dc/db = 1.0
```

### 2. **Topological Sorting**
Uses topological ordering to ensure gradients flow correctly through the computational graph from output to inputs.

**Algorithm**: 
- Performs depth-first search (DFS) to build reverse topological order
- Ensures parent nodes receive gradients before children
- Handles arbitrary DAG (Directed Acyclic Graph) structures

### 3. **Gradient Accumulation**
Implements the chain rule for gradient computation:

```
∂L/∂x = Σ (∂L/∂y) × (∂y/∂x)
```

Each operation stores its local gradient function (`grad_fn`) which computes and accumulates gradients to parent nodes.

### 4. **Supported Operations**

#### Scalar Operations (autograd::Value)
- **Addition**: `add(x, y)` → `dL/dx = dL/dout`, `dL/dy = dL/dout`
- **Multiplication**: `mult(x, y)` → `dL/dx = y × dL/dout`, `dL/dy = x × dL/dout`
- **Subtraction**: `sub(x, y)` → `dL/dx = dL/dout`, `dL/dy = -dL/dout`
- **Division**: `div(x, y)` → `dL/dx = (1/y) × dL/dout`, `dL/dy = -(x/y²) × dL/dout`
- **Exponential**: `exp(x)` → `dL/dx = exp(x) × dL/dout`
- **Logarithm**: `log(x)` → `dL/dx = (1/x) × dL/dout`
- **Maximum**: `max(a, b)` → Gradient flows to the larger value

#### Tensor Operations (autograd::Tensor)
- **Matrix Multiplication**: `matmul(A, B)` - Full backpropagation support
- **Dot Product**: `dot(a, b)` - Inner product with gradients
- **Transpose**: `transpose(A)` - Matrix transposition
- **Bias Addition**: `addBias(X, b)` - Broadcasting bias addition

#### Activation Functions
- **ReLU**: `relu(x)` → `dL/dx = dL/dout × (x > 0 ? 1 : 0)`

### 5. **Gradient Verification**
Implements **finite difference gradient checking** to verify analytical gradients:

```cpp
// Numerical gradient approximation
f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
```

This ensures the backpropagation implementation is mathematically correct.

## 🚀 Getting Started

### Prerequisites
- C++17 compatible compiler (GCC, Clang, or MSVC)
- CMake 3.16 or higher

### Building the Project

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build all targets
make

# Run the main demo
./run_test.sh
```

## 💡 Usage Example

```cpp
#include <autograd/value.hpp>
#include <autograd/ops.hpp>
#include <autograd/backward.hpp>

using namespace autograd;

// Create values with requires_grad=true
auto x = std::make_shared<Value>();
x->value = 2.0;
x->requires_grad = true;

auto w = std::make_shared<Value>();
w->value = 3.0;
w->requires_grad = true;

// Forward pass: y = x * w + 5
auto prod = mult(x, w);        // prod = 6.0
auto b = std::make_shared<Value>();
b->value = 5.0;
auto y = add(prod, b);         // y = 11.0

// Backward pass: compute gradients
backward(y);

// Gradients are now available
std::cout << "dy/dx = " << x->grad << std::endl;  // 3.0
std::cout << "dy/dw = " << w->grad << std::endl;  // 2.0
```

## 🔬 Technical Details

- **Language**: C++17
- **Build System**: CMake
- **Memory Management**: `std::shared_ptr` for automatic memory management
- **Computational Graph**: Dynamic, tape-based recording
- **Gradient Storage**: In-place gradient accumulation in Value nodes
- **Safety Features**: Address and undefined behavior sanitizers enabled

## 📚 Learning Resources

This project is built following the principles from:
- DEEP LEARNING lan Goodfellow, Yoshua Bengio, and Aaron Courville (Ch 6)
- https://www.youtube.com/watch?v=MswxJw-8PvE&t=408s 
- https://www.youtube.com/watch?v=i94OvYb6noo 

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new operations
- Implement additional activation functions
- Improve performance
- Add more comprehensive tests
- Enhance documentation

## 📄 License

See the [LICENSE](LICENSE) file for details.

## 🎯 Future Enhancements

- [ ] GPU acceleration support
- [ ] More activation functions (sigmoid, tanh, softmax)
- [ ] Optimizer implementations (SGD, Adam)
- [ ] Conv2D and pooling layers
- [ ] Loss functions (MSE, CrossEntropy)
- [ ] Model serialization
- [ ] Python bindings
- [ ] File IO

---

