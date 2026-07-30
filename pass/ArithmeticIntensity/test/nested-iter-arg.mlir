// RUN: arithmetic_intensity.py %s | %filecheck %s

// Inner loop bound is a loop-carried iter_arg of the outer loop. The pass
// should trace the iter_arg back to its init value and emit a bandwidth
// equation in both function parameters that drive the loops: outer trip count
// args[1], inner trip count args[2] (from iter_arg init), per-iter bytes 1024.
// CHECK-LABEL: tt.func @nested_iter_arg(
// CHECK-SAME:  %arg0: !tt.ptr<f32> {tt.bandwidth = "args[1] * (args[2] * 1024)"}
tt.func @nested_iter_arg(%in: !tt.ptr<f32>, %M: i32, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %r = scf.for %i = %c0 to %M step %c1
      iter_args(%bound = %N) -> (i32) : i32 {
    scf.for %j = %c0 to %bound step %c1 : i32 {
      %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
    }
    scf.yield %bound : i32
  }
  tt.return
}
