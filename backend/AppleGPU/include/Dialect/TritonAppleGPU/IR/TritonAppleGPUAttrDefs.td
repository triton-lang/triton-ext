#ifndef TRITON_APPLE_GPU_ATTRDEFS
#define TRITON_APPLE_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "TritonAppleGPUDialect.td"

// Lane T and reg R scatter into row/col bits; bases are in
// LinearLayoutConversions.cpp and must match simdgroup_matrix per-lane storage.
//   row = ((T>>1)&1) | ((T>>2)&1)<<1 | ((T>>4)&1)<<2
//   col = R          | ((T>>0)&1)<<1 | ((T>>3)&1)<<2
//
// Spelled as a bare AttrDef with its traits listed, where upstream encodings
// use the `DistributedEncoding` helper: that helper derives from
// `TritonGPU_Attr`, which hardcodes `AttrDef<TritonGPU_Dialect, ...>` and would
// put this attribute in their dialect.
def AppleMmaEncodingAttr : AttrDef<TritonAppleGPU_Dialect, "AppleMmaEncoding",
    [DeclareAttrInterfaceMethods<DistributedEncodingTrait>,
     DeclareAttrInterfaceMethods<MmaEncodingTrait>,
     LayoutEncodingTrait]> {

  let mnemonic = "apple_mma";

  let summary = "Apple simdgroup matmul encoding";

  let description = [{
    Encoding for tensors distributed across Apple simdgroup_multiply_accumulate
    8x8 tiles. Each simdgroup (32 threads) owns one 8x8 output tile.
    warpsPerCTA controls how many simdgroups tile the full tensor.
  }];

  let parameters = (ins
    ArrayRefParameter<"unsigned">:$warpsPerCTA
  );

  let assemblyFormat = "`<` `{` `warpsPerCTA` `=` `[` $warpsPerCTA `]` `}` `>`";

  let extraClassDeclaration = [{
    unsigned getRank() const { return getWarpsPerCTA().size(); }

    mlir::triton::gpu::CGAEncodingAttr getCGALayout() const {
      return mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(getContext(), getRank());
    }
  }];

  let genVerifyDecl = 1;
}

#endif // TRITON_APPLE_GPU_ATTRDEFS
