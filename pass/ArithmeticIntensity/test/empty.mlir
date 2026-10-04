// RUN: arithmetic_intensity.py %s | %filecheck %s

// A function with no memory traffic should not gain any attributes.
// CHECK-LABEL: tt.func @empty(
// CHECK-NOT:   tai.load_bytes
// CHECK-NOT:   tai.store_bytes
// CHECK-NOT:   tai.op_count
tt.func @empty(%p: !tt.ptr<f32>) {
  tt.return
}
