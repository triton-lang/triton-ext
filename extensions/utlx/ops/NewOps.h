#ifndef UTLX_OPS_NEW_OPS_H
#define UTLX_OPS_NEW_OPS_H

#include "triton/Tools/PluginUtils.h"
#include <vector>

namespace utlx {

// TTG ops
void createRemoteShmemStore(TritonOpBuilder &, std::vector<mlir::Value> &);
void createAsyncRemoteShmemStore(TritonOpBuilder &, std::vector<mlir::Value> &);
void createClock64(TritonOpBuilder &, std::vector<mlir::Value> &);

// TTNG ops
void createAsyncStore(TritonOpBuilder &, std::vector<mlir::Value> &);
void createFence(TritonOpBuilder &, std::vector<mlir::Value> &);
void createMapToRemoteBuffer(TritonOpBuilder &, std::vector<mlir::Value> &);
void createClusterSize1D(TritonOpBuilder &, std::vector<mlir::Value> &);
void createAsyncCLCTryCancel(TritonOpBuilder &, std::vector<mlir::Value> &);
void createCLCQueryCancel(TritonOpBuilder &, std::vector<mlir::Value> &);
void createVoteBallotSync(TritonOpBuilder &, std::vector<mlir::Value> &);
void createAsyncTMAPrefetch(TritonOpBuilder &, std::vector<mlir::Value> &);
void createNamedBarrierArrive(TritonOpBuilder &, std::vector<mlir::Value> &);
void createNamedBarrierWait(TritonOpBuilder &, std::vector<mlir::Value> &);

// AMD ops
void createReadBarrierPhase(TritonOpBuilder &, std::vector<mlir::Value> &);

// Modified ops (wrappers for extended signatures)
void createFpToFpWithRbits(TritonOpBuilder &, std::vector<mlir::Value> &);
void createMakeTensorDescWithDescPtr(TritonOpBuilder &,
                                     std::vector<mlir::Value> &);

// Combined require-layout ops (create encoding + RequireLayoutOp in one step)
void createRequireNvMmaSharedLayout(TritonOpBuilder &,
                                    std::vector<mlir::Value> &);
void createRequireNvMmaLayout(TritonOpBuilder &, std::vector<mlir::Value> &);
void createRequireDotOperandLayout(TritonOpBuilder &,
                                   std::vector<mlir::Value> &);
void createRequireTensorMemoryLayout(TritonOpBuilder &,
                                     std::vector<mlir::Value> &);
void createRequireTensorMemoryScalesLayout(TritonOpBuilder &,
                                           std::vector<mlir::Value> &);

// Memory ops
void createAsyncLoad(TritonOpBuilder &, std::vector<mlir::Value> &);
void createGlobalScratchAlloc(TritonOpBuilder &, std::vector<mlir::Value> &);
void createMakeDummyRegisterLayout(TritonOpBuilder &,
                                   std::vector<mlir::Value> &);
void createRequireWithLayoutCarrier(TritonOpBuilder &,
                                    std::vector<mlir::Value> &);
void createAllocClcResponses(TritonOpBuilder &, std::vector<mlir::Value> &);
void createClcQuery(TritonOpBuilder &, std::vector<mlir::Value> &);

// Thread/cluster ops
void createClusterCtaRank(TritonOpBuilder &, std::vector<mlir::Value> &);
void createThreadId(TritonOpBuilder &, std::vector<mlir::Value> &);

} // namespace utlx

#endif // UTLX_OPS_NEW_OPS_H
