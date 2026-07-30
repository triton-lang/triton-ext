// RUN: arithmetic_intensity.py %s | %filecheck %s

// Inner loop bound IS an outer induction variable. The pass substitutes the IV
// with the outer loop's upper bound, yielding a quadratic upper-bound estimate:
// args[1] appears twice (outer trip * IV-substituted inner trip), per-iter 1024.
// CHECK-LABEL: tt.func @triangular(
// CHECK-SAME:  %arg0: !tt.ptr<f32> {tt.bandwidth = "args[1] * (args[1] * 1024)"}
tt.func @triangular(%in: !tt.ptr<f32>, %M: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %M step %c1 : i32 {
    scf.for %j = %c0 to %i step %c1 : i32 {
      %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
    }
  }
  tt.return
}
