#!/usr/bin/env bash
set -euo pipefail

# -------- Configuration --------
BUILD_DIR=build

EXECUTABLES=(
  autograd
  test_value_basic
  test_value_arithmetic
  test_value_advanced
  test_tensor_ops
)
# --------------------------------

echo "🛠  Configuring project..."
cmake -S . -B ${BUILD_DIR}

echo "🔨 Building..."
cmake --build ${BUILD_DIR}

echo "🚀 Running executables..."
echo "--------------------------------"

for exe in "${EXECUTABLES[@]}"; do
  echo "▶ Running ${exe}"
  ./${BUILD_DIR}/${exe}
  echo "✔ ${exe} passed"
  echo "--------------------------------"
done

echo "🎉 All executables ran successfully!"
