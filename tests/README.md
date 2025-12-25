# Autograd C++ Tests

This directory contains test files for the autograd C++ library.

## Test Files

### `test_value_basic.cpp`
Tests basic autograd functionality:
- **Test 1: Leaf** - Tests gradient computation for a simple leaf node
- **Test 2: Addition** - Tests addition operation and gradient propagation
- **Test 3: Chain rule** - Tests chain rule with repeated addition

### `test_value_arithmetic.cpp`
Tests arithmetic operations:
- **Multiplication** - Tests multiplication and its gradients
- **Chain multiplication** - Tests repeated multiplication
- **Division** - Tests division operation
- **Chain division** - Tests repeated division

### `test_value_advanced.cpp`
Tests advanced operations:
- **Exponential** - Tests exp() function and its gradient
- **Softmax-like computation** - Tests a complex computation involving exp, division, and addition (similar to softmax)

### `test_tensor_ops.cpp`
Tests tensor operations:
- **Tensor Dot Product** - Tests matrix dot product and gradient computation for tensors

## Building and Running Tests

### Build all tests:
```bash
cd build
make
```

### Run individual tests:
```bash
./build/test_value_basic
./build/test_value_arithmetic
./build/test_value_advanced
./build/test_tensor_ops
```

### Run all tests:
```bash
cd build
./test_value_basic && ./test_value_arithmetic && ./test_value_advanced && ./test_tensor_ops
```

## Expected Output

All tests print their results with expected values in parentheses. Gradients should match the expected values (within numerical precision for floating-point operations).

Example output from `test_value_basic`:
```
=== Test 1: Leaf ===
x.grad = 1 (expected 1)
```

## Notes

- All tests use the `backward()` function for automatic differentiation
- Tests verify both forward (value computation) and backward (gradient computation) passes
- The library uses address and undefined behavior sanitizers during development for bug detection
