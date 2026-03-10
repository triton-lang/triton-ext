// RUN: triton-opt --split-input-file %s -canonicalize | FileCheck %s

tt.func @example_zero() -> tensor<4x4x8xf16> {
  %t = example.zero -> tensor<4x4x8xf16>
  // CHECK: example.zero
  tt.return %t
}
