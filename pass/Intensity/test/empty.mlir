// RUN: intensity.py %s | %filecheck %s

// A function with no memory traffic should not gain any attributes.
// CHECK-LABEL: tt.func @empty(
// CHECK-NOT:   tint.load_bytes
// CHECK-NOT:   tint.store_bytes
// CHECK-NOT:   tint.op_count
tt.func @empty(%p: !tt.ptr<f32>) {
  tt.return
}
