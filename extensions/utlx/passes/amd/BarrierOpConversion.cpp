/// AMD Barrier Op LLVM Conversion Patterns ported to uTLX plugin.
///
/// Provides LLVM lowering patterns for AMD barrier ops:
///   - InitBarrierOp -> store count/phase to shared memory
///   - ArriveBarrierOp -> ds_dec_rtn_u32 with conditional phase flip
///   - ReadBarrierPhaseOp -> load phase from shared memory
///
/// Ported from third_party/amd/lib/TritonAMDGPUToLLVM/BarrierOpConversion.cpp

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

// AMD dialect includes
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"

using namespace mlir;

namespace {

Value getBarrierField(triton::TritonLLVMOpBuilder builder,
                      SharedMemoryObject barrierSmemObj, int fieldIndex) {
  OpBuilder b = *builder.builder;
  auto i32ty = b.getIntegerType(32);
  auto baseAddr = barrierSmemObj.getBase();
  return builder.gep(baseAddr.getType(), i32ty, baseAddr,
                     builder.i32_val(fieldIndex));
}

Value getPhaseBaseAddress(TritonLLVMOpBuilder builder,
                          SharedMemoryObject barrierSmemObj) {
  return getBarrierField(builder, barrierSmemObj, 1);
}

Value getCountBaseAddress(TritonLLVMOpBuilder builder,
                          SharedMemoryObject barrierSmemObj) {
  return getBarrierField(builder, barrierSmemObj, 0);
}

struct UTLXInitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto barrierSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    int count = op.getCount();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto countBaseAddr = getCountBaseAddress(b, barrierSmemObj);
    auto phaseBaseAddr = getPhaseBaseAddress(b, barrierSmemObj);
    Value countVal = b.i32_val(count - 1);
    Value phaseVal = b.i32_val(0);
    b.store(countVal, countBaseAddr);
    b.store(phaseVal, phaseBaseAddr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct UTLXArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto barrierSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    int decCount = op.getCount();
    assert(decCount == 1 && "Only support decCount == 1 for now");

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto countBaseAddr = getCountBaseAddress(b, barrierSmemObj);
    auto phaseBaseAddr = getPhaseBaseAddress(b, barrierSmemObj);

    auto resetVal = b.i32_val(op.getExpectedCount() - 1);
    GCNBuilder gcnBuilder;
    auto &dec_rtn = *gcnBuilder.create("ds_dec_rtn_u32");
    auto retVal = gcnBuilder.newOperand("=v");
    auto countBaseAddrArg = gcnBuilder.newOperand(countBaseAddr, "v");
    auto resetValArg = gcnBuilder.newOperand(resetVal, "v");
    dec_rtn(retVal, countBaseAddrArg, resetValArg);
    auto &wait_cnt = *gcnBuilder.create("s_waitcnt lgkmcnt(0)");
    wait_cnt();
    auto preDecrementCountVal = gcnBuilder.launch(rewriter, loc, i32_ty, true);

    MLIRContext *ctx = rewriter.getContext();
    Value zero = b.i32_val(0);
    Value allArrived = b.icmp_eq(preDecrementCountVal, zero);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterPhaseFlipBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *phaseFlipBlock = rewriter.createBlock(afterPhaseFlipBlock);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, allArrived, phaseFlipBlock,
                                    afterPhaseFlipBlock);

    rewriter.setInsertionPointToStart(phaseFlipBlock);
    GCNBuilder phaseFlipBuilder;
    auto &xor_phase = *phaseFlipBuilder.create("ds_xor_b32");
    auto baseAddrArg = phaseFlipBuilder.newOperand(phaseBaseAddr, "v");
    Value one = b.i32_val(1);
    auto oneArg = phaseFlipBuilder.newOperand(one, "v");
    xor_phase(baseAddrArg, oneArg);

    auto &s_wakeup = *phaseFlipBuilder.create("s_wakeup");
    s_wakeup();
    phaseFlipBuilder.launch(rewriter, loc, void_ty(ctx), true);

    rewriter.create<LLVM::BrOp>(loc, afterPhaseFlipBlock);
    rewriter.eraseOp(op);
    return success();
  }
};

struct UTLXReadBarrierPhaseOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ReadBarrierPhaseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::ReadBarrierPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto barrierSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto phaseBaseAddr = getPhaseBaseAddress(b, barrierSmemObj);
    auto res = b.load(i32_ty, phaseBaseAddr);

    GCNBuilder phaseReadBuilder;
    MLIRContext *ctx = rewriter.getContext();
    auto &wait_cnt = *phaseReadBuilder.create("s_waitcnt lgkmcnt(0)");
    wait_cnt();
    phaseReadBuilder.launch(rewriter, loc, void_ty(ctx), true);
    rewriter.replaceOp(op, res);
    return success();
  }
};

} // namespace

namespace utlx {

void populateAMDBarrierOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns,
                                        mlir::PatternBenefit benefit) {
  patterns.add<UTLXInitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<UTLXArriveBarrierOpConversion>(typeConverter, benefit);
  patterns.add<UTLXReadBarrierPhaseOpConversion>(typeConverter, benefit);
}

} // namespace utlx
