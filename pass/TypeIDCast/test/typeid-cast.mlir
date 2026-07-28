// An extension pass can cast<triton::FuncOp> under triton-ext-opt.
//
// RUN: triton-ext-opt %s --test-triton-typeid-cast | FileCheck %s

// CHECK-LABEL: tt.func public @kernel
// CHECK-SAME: triton_ext.typeid_cast_ok
module {
  tt.func public @kernel() {
    tt.return
  }
}
