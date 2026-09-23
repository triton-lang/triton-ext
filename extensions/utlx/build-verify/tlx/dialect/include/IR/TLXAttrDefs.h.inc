/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::triton::tlx {

/// Placeholder layout for register-distributed tensors to be resolved after inlining
/// This attribute represents a placeholder layout for tensors distributed
/// across registers. It is generated during initial lowering when we don't
/// have enough context to determine the final distribution layout.
/// 
/// After function inlining, a pass will resolve this to a concrete layout such as:
/// - BlockedEncodingAttr (default blocked distribution)
/// - TMEM-compatible BlockedEncodingAttr (for tensors loaded from TMEM)
/// - MmaEncodingAttr (for MMA operation results)
/// - DotOperandEncodingAttr (for dot operation inputs)
/// 
/// Parameters:
/// - shape: The shape of the tensor
/// - elementType: The element type
/// - tmemCompatible: If true, create a layout compatible with TMEM load/store
class DummyRegisterLayoutAttr;
/// Placeholder layout for TMEM tensors to be resolved during layout propagation
/// This attribute represents a placeholder layout for tensors in Tensor Memory (TMEM).
/// It is used when we don't know the final TMEM layout at allocation time.
/// 
/// During layout propagation, this will be resolved to a concrete TMEM layout:
/// - TensorMemoryEncodingAttr (for regular TMEM data)
/// - TensorMemoryScalesEncodingAttr (for scales in scaled MMA operations)
/// 
/// The resolution depends on how the TMEM buffer is used (e.g., as scales in tmem_copy).
class DummyTMEMLayoutAttr;
namespace detail {

struct DummyRegisterLayoutAttrStorage;

} // namespace detail
class DummyRegisterLayoutAttr : public ::mlir::Attribute::AttrBase<DummyRegisterLayoutAttr, ::mlir::Attribute, detail::DummyRegisterLayoutAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tlx.dummy_register_layout";
  static constexpr ::llvm::StringLiteral dialectName = "tlx";
  static DummyRegisterLayoutAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType, bool tmemCompatible);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dummy_register_layout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  bool getTmemCompatible() const;
};
class DummyTMEMLayoutAttr : public ::mlir::Attribute::AttrBase<DummyTMEMLayoutAttr, ::mlir::Attribute, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tlx.dummy_tmem_layout";
  static constexpr ::llvm::StringLiteral dialectName = "tlx";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dummy_tmem_layout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::DummyRegisterLayoutAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::DummyTMEMLayoutAttr)

#endif // GET_ATTRDEF_CLASSES

