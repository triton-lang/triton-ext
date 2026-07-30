// RUN: arithmetic_intensity.py %s | %filecheck %s

// A function with no memory traffic should not gain any attributes.
// CHECK-LABEL: tt.func @empty(
// CHECK-NOT:   tt.bandwidth
// CHECK-NOT:   tt.compute
tt.func @empty(%p: !tt.ptr<f32>) {
  tt.return
}
