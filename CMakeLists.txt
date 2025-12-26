cmake_minimum_required(VERSION 3.16)

project(autograd_cpp
    VERSION 0.1
    LANGUAGES CXX
)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create a library for the autograd components
add_library(autograd_lib
    src/value.cpp
    src/ops.cpp
    src/backward.cpp
    src/graph_utils.cpp
    src/tensor.cpp
    src/constant.cpp
    src/activations.cpp
)

# Include directories for the library
target_include_directories(autograd_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Warnings (good C++ hygiene)
target_compile_options(autograd_lib PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Uncomment these while developing to catch bugs early
target_compile_options(autograd_lib PRIVATE -fsanitize=address,undefined)
target_link_options(autograd_lib PRIVATE -fsanitize=address,undefined)

# Build the main executable
add_executable(autograd
    src/main.cpp
)

target_link_libraries(autograd PRIVATE autograd_lib)

# Build test executables
add_executable(test_value_basic
    tests/test_value_basic.cpp
)
target_link_libraries(test_value_basic PRIVATE autograd_lib)
target_compile_options(test_value_basic PRIVATE -fsanitize=address,undefined)
target_link_options(test_value_basic PRIVATE -fsanitize=address,undefined)

add_executable(test_value_arithmetic
    tests/test_value_arithmetic.cpp
)
target_link_libraries(test_value_arithmetic PRIVATE autograd_lib)
target_compile_options(test_value_arithmetic PRIVATE -fsanitize=address,undefined)
target_link_options(test_value_arithmetic PRIVATE -fsanitize=address,undefined)

add_executable(test_value_advanced
    tests/test_value_advanced.cpp
)
target_link_libraries(test_value_advanced PRIVATE autograd_lib)
target_compile_options(test_value_advanced PRIVATE -fsanitize=address,undefined)
target_link_options(test_value_advanced PRIVATE -fsanitize=address,undefined)

add_executable(test_tensor_ops
    tests/test_tensor_ops.cpp
)
target_link_libraries(test_tensor_ops PRIVATE autograd_lib)
target_compile_options(test_tensor_ops PRIVATE -fsanitize=address,undefined)
target_link_options(test_tensor_ops PRIVATE -fsanitize=address,undefined)

add_executable(test_activations
    tests/test_activations.cpp
)
target_link_libraries(test_activations PRIVATE autograd_lib)
target_compile_options(test_activations PRIVATE -fsanitize=address,undefined)
target_link_options(test_activations PRIVATE -fsanitize=address,undefined)

add_executable(test_nn
    tests/test_nn.cpp
)
target_link_libraries(test_nn PRIVATE autograd_lib)
target_compile_options(test_nn PRIVATE -fsanitize=address,undefined)
target_link_options(test_nn PRIVATE -fsanitize=address,undefined)


add_executable(test_bias
    tests/test_bias.cpp
)
target_link_libraries(test_bias PRIVATE autograd_lib)
target_compile_options(test_bias PRIVATE -fsanitize=address,undefined)
target_link_options(test_bias PRIVATE -fsanitize=address,undefined)

add_executable(test_finite_diff
    tests/test_finite_diffference_gradient.cpp
)
target_link_libraries(test_finite_diff PRIVATE autograd_lib)
target_compile_options(test_finite_diff PRIVATE -fsanitize=address,undefined)
target_link_options(test_finite_diff PRIVATE -fsanitize=address,undefined)