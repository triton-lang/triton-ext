// RUN: arithmetic_intensity.py %s | %filecheck %s

// A loop with a constant trip count folds to a literal byte count. This case
// keeps the `tensor<256x!tt.ptr<f32>>` parameter shape (and threads the pointer
// tensor through `scf.for` `iter_args`) so we keep regression coverage of
// `findPointerParam` walking back through a loop-carried tensor-of-pointers to
// its function-arg base.
//
// 4 iterations * 256 elements * 4 bytes/elem = 4096 bytes per pointer.
// CHECK-LABEL: tt.func @const_loop(
// CHECK-SAME:  %arg0: tensor<256x!tt.ptr<f32>> {tt.bandwidth = "4096", tt.compute = "0"}
// CHECK-SAME:  %arg1: tensor<256x!tt.ptr<f32>> {tt.bandwidth = "4096"}
tt.func @const_loop(%out: tensor<256x!tt.ptr<f32>>,
                    %in: tensor<256x!tt.ptr<f32>>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %off = tt.splat %c1 : i32 -> tensor<256xi32>
  %r:2 = scf.for %i = %c0 to %c4 step %c1
      iter_args(%pin = %in, %pout = %out)
      -> (tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>) : i32 {
    %v = tt.load %pin : tensor<256x!tt.ptr<f32>>
    tt.store %pout, %v : tensor<256x!tt.ptr<f32>>
    %nin = tt.addptr %pin, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %nout = tt.addptr %pout, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %nin, %nout : tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
  }
  tt.return
}
