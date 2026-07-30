// RUN: arithmetic_intensity.py %s | %filecheck %s

// The dynamic loop range becomes a symbol on the function arg. Uses the typical
// kernel idiom: scalar `!tt.ptr<f32>` parameters with the address tensor built
// from `tt.make_range` + `tt.splat` + `tt.addptr`.
//
// Per-iter bytes = 256 * 4 = 1024, trip count = args[2].
// CHECK-LABEL: tt.func @param_loop(
// CHECK-SAME:  %arg0: !tt.ptr<f32> {tt.bandwidth = "args[2] * 1024", tt.compute = "0"}
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tt.bandwidth = "args[2] * 1024"}
tt.func @param_loop(%out: !tt.ptr<f32>,
                    %in: !tt.ptr<f32>,
                    %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %outb = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %outp = tt.addptr %outb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %N step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
    tt.store %outp, %v : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}
