/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TLXDialect.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::TLXDialect)
namespace mlir::triton::tlx {


TLXDialect::TLXDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TLXDialect>())
    
     {
  getContext()->loadDialect<triton::TritonDialect>();
  getContext()->loadDialect<triton::gpu::TritonGPUDialect>();
  initialize();
}

TLXDialect::~TLXDialect() = default;


} // namespace mlir::triton::tlx
