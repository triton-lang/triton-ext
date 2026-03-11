// RUN: triton-opt --split-input-file %s -canonicalize | FileCheck %s

// This test checks that the `example.zero` op is correctly parsed and printed
// and that it is not canonicalized away. In a real dialect, we would want to
// check that operations are correctly lowered, but this example dialect does
// not include any such pass.
tt.func @example_zero() -> !example.tile
{
  %t = example.zero -> !example.tile
  // CHECK: example.zero
  tt.return %t: !example.tile
}
