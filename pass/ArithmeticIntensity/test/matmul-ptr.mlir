// RUN: arithmetic_intensity.py %s | %filecheck %s

// Block-matmul kernel with scalar `!tt.ptr<f16>` parameters. Builds 64x64
// address tensors from `tt.make_range` + `tt.expand_dims` + `tt.broadcast` +
// `tt.splat` + `tt.addptr` (the usual Triton address arithmetic), iterates the
// K loop `K/64` times, accumulates a `tt.dot`, then truncs to f16 and stores
// once. The pass should:
//   * attribute A/B bandwidth = 8192 * (K/64) (per-iter tensor<64x64xf16> =
//     8192 bytes, times the loop trip count).
//   * attribute C bandwidth = 8192 (single out-of-loop store).
//   * attribute C compute = dot FLOPs scaled by trip count (2*M*N*K = 524288
//     per dot block with M=N=K=64) plus the epilogue truncf (64*64 = 4096).
// CHECK-LABEL: tt.func @matmul_ptr(
// CHECK-SAME:  %arg0: !tt.ptr<f16> {tt.bandwidth = "(args[5] / 64) * 8192"}
// CHECK-SAME:  %arg1: !tt.ptr<f16> {tt.bandwidth = "(args[5] / 64) * 8192"}
// CHECK-SAME:  %arg2: !tt.ptr<f16> {tt.bandwidth = "8192", tt.compute = "(args[5] / 64) * 524288 + 4096"}
tt.func @matmul_ptr(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f16>,
                    %M: i32, %N: i32, %K: i32,
                    %sam: i32, %sak: i32,
                    %sbk: i32, %sbn: i32,
                    %scm: i32, %scn: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %zero = arith.constant 0.0 : f32
  %acc0 = tt.splat %zero : f32 -> tensor<64x64xf32>
  %rm = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %rn = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %rk = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %rm_col = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %rn_row = tt.expand_dims %rn {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %rk_row = tt.expand_dims %rk {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %rk_col = tt.expand_dims %rk {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %sam_t = tt.splat %sam : i32 -> tensor<64x1xi32>
  %sak_t = tt.splat %sak : i32 -> tensor<1x64xi32>
  %a_m = arith.muli %rm_col, %sam_t : tensor<64x1xi32>
  %a_k = arith.muli %rk_row, %sak_t : tensor<1x64xi32>
  %a_m_b = tt.broadcast %a_m : tensor<64x1xi32> -> tensor<64x64xi32>
  %a_k_b = tt.broadcast %a_k : tensor<1x64xi32> -> tensor<64x64xi32>
  %a_off = arith.addi %a_m_b, %a_k_b : tensor<64x64xi32>
  %ab = tt.splat %A : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
  %ap = tt.addptr %ab, %a_off : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
  %sbk_t = tt.splat %sbk : i32 -> tensor<64x1xi32>
  %sbn_t = tt.splat %sbn : i32 -> tensor<1x64xi32>
  %b_k = arith.muli %rk_col, %sbk_t : tensor<64x1xi32>
  %b_n = arith.muli %rn_row, %sbn_t : tensor<1x64xi32>
  %b_k_b = tt.broadcast %b_k : tensor<64x1xi32> -> tensor<64x64xi32>
  %b_n_b = tt.broadcast %b_n : tensor<1x64xi32> -> tensor<64x64xi32>
  %b_off = arith.addi %b_k_b, %b_n_b : tensor<64x64xi32>
  %bb = tt.splat %B : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
  %bp = tt.addptr %bb, %b_off : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
  %final = scf.for %k = %c0 to %K step %c64
      iter_args(%acc = %acc0) -> (tensor<64x64xf32>) : i32 {
    %a = tt.load %ap : tensor<64x64x!tt.ptr<f16>>
    %b = tt.load %bp : tensor<64x64x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %acc :
        tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    scf.yield %d : tensor<64x64xf32>
  }
  %scm_t = tt.splat %scm : i32 -> tensor<64x1xi32>
  %scn_t = tt.splat %scn : i32 -> tensor<1x64xi32>
  %c_m = arith.muli %rm_col, %scm_t : tensor<64x1xi32>
  %c_n = arith.muli %rn_row, %scn_t : tensor<1x64xi32>
  %c_m_b = tt.broadcast %c_m : tensor<64x1xi32> -> tensor<64x64xi32>
  %c_n_b = tt.broadcast %c_n : tensor<1x64xi32> -> tensor<64x64xi32>
  %c_off = arith.addi %c_m_b, %c_n_b : tensor<64x64xi32>
  %cb = tt.splat %C : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
  %cp = tt.addptr %cb, %c_off : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
  %final_f16 = arith.truncf %final : tensor<64x64xf32> to tensor<64x64xf16>
  tt.store %cp, %final_f16 : tensor<64x64x!tt.ptr<f16>>
  tt.return
}
