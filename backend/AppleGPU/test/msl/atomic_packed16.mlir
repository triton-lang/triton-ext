#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#loc = loc("atomic_packed16.py":3:1)
#loc6 = loc("P"(#loc))
#loc7 = loc("X"(#loc))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "mps:apple_m", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atom_packed16(%P: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("P"(#loc)), %X: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("X"(#loc))) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<64xi1, #blocked> loc(#loc1)
    %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1> loc(#loc8)
    %0 = tt.splat %P : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked> loc(#loc3)
    %1 = tt.splat %X : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked1> loc(#loc4)
    %2 = tt.addptr %1, %n : tensor<64x!tt.ptr<f16>, #blocked1>, tensor<64xi32, #blocked1> loc(#loc4)
    %3 = tt.load %2 : tensor<64x!tt.ptr<f16>, #blocked1> loc(#loc5)
    %4 = ttg.convert_layout %3 : tensor<64xf16, #blocked1> -> tensor<64xf16, #blocked> loc(#loc1)
    %5 = tt.atomic_rmw fadd, acq_rel, gpu, %0, %4, %cst : (tensor<64x!tt.ptr<f16>, #blocked>, tensor<64xf16, #blocked>, tensor<64xi1, #blocked>) -> tensor<64xf16, #blocked> loc(#loc1)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("atomic_packed16.py":5:5)
#loc2 = loc("atomic_packed16.py":4:9)
#loc3 = loc("atomic_packed16.py":5:19)
#loc4 = loc("atomic_packed16.py":5:38)
#loc5 = loc("atomic_packed16.py":5:30)
#loc8 = loc("n"(#loc2))
