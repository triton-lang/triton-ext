// Simdgroup MMA encoding to LinearLayout. For lane T and element e:
//   col = e | ((T&1)<<1) | (((T>>3)&1)<<2)
//   row = ((T>>1)&1) | (((T>>2)&1)<<1) | (((T>>4)&1)<<2)

#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

#define S(v) StringAttr::get(ctx, (v))

namespace mlir::triton::applegpu {

LinearLayout
AppleMmaEncodingAttr::toLinearLayout(llvm::ArrayRef<int64_t> shape) const {

  MLIRContext *ctx = getContext();
  int rank = shape.size();
  assert((rank == 2 || rank == 3) && "AppleMmaEncoding is 2D or batched 2D");

  auto dimNames = standardOutDimNames(ctx, rank);

  // Rank 3 prepends a batch axis the hardware tile never touches: lane and
  // warp bases stay in row/col, so `combineCtaCgaWithShape` covers the batch
  // extent with register reps and each register keeps a fixed slice index.
  const auto rc = [&](int32_t row, int32_t col) {
    std::vector<int32_t> v;
    if (rank == 3)
      v.push_back(0);
    v.push_back(row);
    v.push_back(col);
    return v;
  };

  // Must match simdgroup_matrix per-lane storage.
  std::vector<std::vector<int32_t>> registerBases{rc(0, 1)};
  std::vector<std::vector<int32_t>> laneBases{
      rc(0, 2), // L0 -> col bit1
      rc(1, 0), // L1 -> row bit0
      rc(2, 0), // L2 -> row bit1
      rc(0, 4), // L3 -> col bit2
      rc(4, 0), // L4 -> row bit2
  };
  LinearLayout ctaLayout(
      SmallVector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>{
          {S(lldim::Register), registerBases}, {S(lldim::Lane), laneBases}},
      dimNames);

  auto wpc = getWarpsPerCTA();
  assert((int)wpc.size() == rank);

  // Column-major warp tiling: warpId % wN gives col, warpId / wN gives row.
  // The rank-3 batch entry is 1 and contributes no warp bits.
  SmallVector<unsigned> warpOrder;
  for (int i = rank - 1; i >= 0; --i)
    warpOrder.push_back((unsigned)i);
  ctaLayout *= identityStandardND(S(lldim::Warp), wpc, warpOrder)
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCGALayout(), shape);
}

} // namespace mlir::triton::applegpu
