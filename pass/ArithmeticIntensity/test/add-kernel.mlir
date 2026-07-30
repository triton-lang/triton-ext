// RUN: arithmetic_intensity.py %s | %filecheck %s

// Stored values report a compute metric in addition to bandwidth. Loaded arrays
// get bandwidth; the stored array additionally accumulates the chain's compute
// (one elementwise addf over 256 elements = 256 ops).
// CHECK-LABEL: tt.func @add_kernel(
// CHECK-SAME:  %arg0: !tt.ptr<f32> {tt.bandwidth = "1024", tt.compute = "256"}
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tt.bandwidth = "1024"}
// CHECK-SAME:  %arg2: !tt.ptr<f32> {tt.bandwidth = "1024"}
tt.func @add_kernel(%out: !tt.ptr<f32>,
                    %a: !tt.ptr<f32>,
                    %b: !tt.ptr<f32>) {
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %ab = tt.splat %a : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %bb = tt.splat %b : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %ob = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %ap = tt.addptr %ab, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %bp = tt.addptr %bb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %op = tt.addptr %ob, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %va = tt.load %ap : tensor<256x!tt.ptr<f32>>
  %vb = tt.load %bp : tensor<256x!tt.ptr<f32>>
  %s = arith.addf %va, %vb : tensor<256xf32>
  tt.store %op, %s : tensor<256x!tt.ptr<f32>>
  tt.return
}
