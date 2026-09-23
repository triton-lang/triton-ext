/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TLXOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::tlx {

/// Create an alias of a local memory buffer
/// Creates an alias of a local memory buffer with a different view (shape,
/// element type, or encoding). This operation is produced during the
/// StorageAliasAllocationPass when lowering StorageAliasLocalAllocOp.
/// 
/// Example:
/// ```mlir
/// %backing = ttg.local_alloc : () -> !ttg.memdesc<32768xi8, #shared, #smem, mutable>
/// %alias = tlx.local_alias %backing
///          : !ttg.memdesc<32768xi8, #shared, #smem, mutable>
///          -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
/// ```
class LocalAliasOp;

} // namespace mlir::triton::tlx
namespace mlir::triton::tlx {

/// release specific layout for a register buffer
class ReleaseLayoutOp;

} // namespace mlir::triton::tlx
namespace mlir::triton::tlx {

/// require specific layout for a local memory buffer
class RequireLayoutOp;

} // namespace mlir::triton::tlx
namespace mlir::triton::tlx {

/// Define a reuse group for buffer overlap relationships
/// Creates a reuse group that defines buffer overlap relationships for
/// memory allocations (shared memory or tensor memory). A reuse group
/// organizes multiple buffers (or nested groups) with a specific
/// relationship type:
/// 
/// - **shared**: Elements logically occupy the same memory region at each
///   buffer index. Useful when buffers are used at different times and can
///   share the same physical memory.
/// - **distinct**: Elements must be placed in non-overlapping memory regions.
///   Useful when buffers need to be accessed simultaneously.
/// 
/// The reuse group forms a tree structure where:
/// - Leaf nodes are `!ttg.memdesc` values (buffered tensors from local_alloc
///   stored in smem or tmem)
/// - Internal nodes are nested `!tlx.reuse_group` values
/// 
/// The `group_size` attribute enables **subtiling** for mixed buffer counts.
/// When group_size > 1, K consecutive buffers are treated as a single logical
/// group for offset calculation. For example, if a tensor has 4 buffers and
/// group_size=2, buffers [0,1] form logical group 0 and [2,3] form group 1.
/// 
/// Note: The storage_alias_spec is NOT part of this operation. Validation
/// that all elements reference the same storage_alias_spec is performed
/// by the SetBufferOverlapOp verifier when the overlap scheme is defined.
/// 
/// Constraints:
/// - At least one element must be provided.
/// - All elements must use the same storage kind (smem or tmem).
/// - group_size must be a positive integer (default: 1).
/// 
/// Example:
/// ```mlir
/// // Simple shared group: A and B share the same memory
/// %group = tlx.reuse_group(%a, %b) group_kind = shared, group_size = 1
///          : (!ttg.memdesc<2x64x64xf32, ...>, !ttg.memdesc<2x64x64xbf16, ...>)
///          -> !tlx.reuse_group<shared>
/// 
/// // Subtiling: P has 4 buffers, treated as 2 logical groups of 2
/// %subtiled = tlx.reuse_group(%p) group_kind = shared, group_size = 2
///             : (!ttg.memdesc<4x64x64xf32, ...>)
///             -> !tlx.reuse_group<shared>
/// ```
class ReuseGroupOp;

} // namespace mlir::triton::tlx
namespace mlir::triton::tlx {

/// Define the buffer overlap scheme for a storage alias spec
/// Defines the buffer overlap scheme for allocations using a storage alias spec.
/// This operation links a storage_alias_spec to its overlap definition (a reuse_group).
/// 
/// The compiler will use this information in subsequent passes to:
/// 1. Validate that the overlap scheme is achievable
/// 2. Compute buffer offsets to satisfy the overlap requirements
/// 
/// This operation is eliminated during the ReusedBufferOffsetCalculationPass
/// after offsets have been computed and applied.
/// 
/// Constraints:
/// - All leaf elements in the reuse_group tree must be allocated from the
///   same storage_alias_spec via tlx.storage_alias_local_alloc
/// - All elements must use the same storage kind (smem or tmem)
/// - This operation should appear after all local_alloc operations that
///   reference the storage_alias_spec
/// 
/// Example:
/// ```mlir
/// %spec = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
/// %qk = tlx.storage_alias_local_alloc %spec : ... -> !ttg.memdesc<...>
/// %p = tlx.storage_alias_local_alloc %spec : ... -> !ttg.memdesc<...>
/// 
/// %group = tlx.reuse_group(%qk, %p) group_kind = shared
///          : (!ttg.memdesc<...>, !ttg.memdesc<...>) -> !tlx.reuse_group<shared>
/// 
/// tlx.set_buffer_overlap(%spec, %group)
///          : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
/// ```
class SetBufferOverlapOp;

} // namespace mlir::triton::tlx
namespace mlir::triton::tlx {

/// Allocate local memory referencing a storage alias specification
/// Allocates local memory (shared memory or tensor memory) that references
/// a storage alias specification. Multiple allocations can reference the same
/// storage alias specification, and the compiler will:
/// 1. Compute the required buffer size (or validate the explicit size)
/// 2. Assign offsets to each allocation
/// 3. Materialize the actual memory allocation
/// 
/// This operation is produced by the Python frontend when `local_alloc` is
/// called with a `storage_alias_spec` in the `reuse` parameter.
/// 
/// After the StorageAliasAllocationPass runs, this operation is replaced with
/// a LocalAliasOp pointing to a standard LocalAllocOp/TMEMAllocOp.
/// 
/// Example:
/// ```mlir
/// %alias = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
/// %buf = tlx.storage_alias_local_alloc %alias : !tlx.storage_alias_spec<smem>
///        -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
/// ```
class StorageAliasLocalAllocOp;

} // namespace mlir::triton::tlx
namespace mlir::triton::tlx {

/// Define a storage alias specification
/// Creates a storage alias specification that can be referenced by multiple
/// `local_alloc` operations. This operation does not allocate memory itself;
/// it defines a logical grouping for buffer sharing.
/// 
/// The actual memory allocation is deferred until `local_alloc` operations
/// reference this storage alias spec. The compiler will:
/// - If `buffer_size_bytes` is specified: verify all references fit within
///   the specified size.
/// - Otherwise: compute the size as the maximum of all referencing allocations.
/// 
/// Note: Only smem and tmem storage kinds are supported. smemCluster is not
/// allowed.
/// 
/// Example:
/// ```mlir
/// %alias = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
/// %alias_sized = tlx.storage_alias_spec storage = tmem, size = 16384 : !tlx.storage_alias_spec<tmem, 16384>
/// ```
class StorageAliasSpecOp;

} // namespace mlir::triton::tlx
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::LocalAliasOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LocalAliasOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LocalAliasOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tlx.local_alias", odsAttrs.getContext());
  }

  LocalAliasOpGenericAdaptorBase(LocalAliasOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LocalAliasOpGenericAdaptor : public detail::LocalAliasOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LocalAliasOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
public:
  LocalAliasOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LocalAliasOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : LocalAliasOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LocalAliasOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LocalAliasOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LocalAliasOpGenericAdaptor(RangeT values, const LocalAliasOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LocalAliasOp, typename = std::enable_if_t<std::is_same_v<LateInst, LocalAliasOp>>>
  LocalAliasOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LocalAliasOpAdaptor : public LocalAliasOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LocalAliasOpGenericAdaptor::LocalAliasOpGenericAdaptor;
  LocalAliasOpAdaptor(LocalAliasOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LocalAliasOp : public ::mlir::Op<LocalAliasOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::triton::gpu::TensorOrMemDesc>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::MemDescViewTrait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LocalAliasOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LocalAliasOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tlx.local_alias");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::gpu::TensorOrMemDesc> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::TensorOrMemDesc>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::gpu::TensorOrMemDesc> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::TensorOrMemDesc>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src);
  static LocalAliasOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src);
  static LocalAliasOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static LocalAliasOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static LocalAliasOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LocalAliasOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LocalAliasOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LocalAliasOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LocalAliasOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::LocalAliasOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::ReleaseLayoutOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ReleaseLayoutOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ReleaseLayoutOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tlx.release_layout", odsAttrs.getContext());
  }

  ReleaseLayoutOpGenericAdaptorBase(ReleaseLayoutOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ReleaseLayoutOpGenericAdaptor : public detail::ReleaseLayoutOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReleaseLayoutOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
public:
  ReleaseLayoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ReleaseLayoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ReleaseLayoutOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ReleaseLayoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ReleaseLayoutOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ReleaseLayoutOpGenericAdaptor(RangeT values, const ReleaseLayoutOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ReleaseLayoutOp, typename = std::enable_if_t<std::is_same_v<LateInst, ReleaseLayoutOp>>>
  ReleaseLayoutOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReleaseLayoutOpAdaptor : public ReleaseLayoutOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReleaseLayoutOpGenericAdaptor::ReleaseLayoutOpGenericAdaptor;
  ReleaseLayoutOpAdaptor(ReleaseLayoutOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ReleaseLayoutOp : public ::mlir::Op<ReleaseLayoutOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultElementType, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReleaseLayoutOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReleaseLayoutOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tlx.release_layout");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src);
  static ReleaseLayoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src);
  static ReleaseLayoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static ReleaseLayoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static ReleaseLayoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReleaseLayoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReleaseLayoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ReleaseLayoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ReleaseLayoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::ReleaseLayoutOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::RequireLayoutOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RequireLayoutOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  RequireLayoutOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tlx.require_layout", odsAttrs.getContext());
  }

  RequireLayoutOpGenericAdaptorBase(RequireLayoutOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class RequireLayoutOpGenericAdaptor : public detail::RequireLayoutOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RequireLayoutOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
public:
  RequireLayoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RequireLayoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : RequireLayoutOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RequireLayoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RequireLayoutOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  RequireLayoutOpGenericAdaptor(RangeT values, const RequireLayoutOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = RequireLayoutOp, typename = std::enable_if_t<std::is_same_v<LateInst, RequireLayoutOp>>>
  RequireLayoutOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RequireLayoutOpAdaptor : public RequireLayoutOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RequireLayoutOpGenericAdaptor::RequireLayoutOpGenericAdaptor;
  RequireLayoutOpAdaptor(RequireLayoutOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RequireLayoutOp : public ::mlir::Op<RequireLayoutOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::triton::gpu::TensorOrMemDesc>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultElementType, ::mlir::OpTrait::MemDescViewTrait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RequireLayoutOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RequireLayoutOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tlx.require_layout");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::gpu::TensorOrMemDesc> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::TensorOrMemDesc>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::gpu::TensorOrMemDesc> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::TensorOrMemDesc>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src);
  static RequireLayoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src);
  static RequireLayoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static RequireLayoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static RequireLayoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RequireLayoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RequireLayoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RequireLayoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RequireLayoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::RequireLayoutOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::ReuseGroupOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ReuseGroupOpGenericAdaptorBase {
public:
  struct Properties {
    using group_kindTy = ::mlir::triton::tlx::ReuseGroupKindAttr;
    group_kindTy group_kind;

    auto getGroupKind() const {
      auto &propStorage = this->group_kind;
      return ::llvm::cast<::mlir::triton::tlx::ReuseGroupKindAttr>(propStorage);
    }
    void setGroupKind(const ::mlir::triton::tlx::ReuseGroupKindAttr &propValue) {
      this->group_kind = propValue;
    }
    using group_sizeTy = ::mlir::IntegerAttr;
    group_sizeTy group_size;

    auto getGroupSize() const {
      auto &propStorage = this->group_size;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setGroupSize(const ::mlir::IntegerAttr &propValue) {
      this->group_size = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.group_kind == this->group_kind &&
        rhs.group_size == this->group_size &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ReuseGroupOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tlx.reuse_group", odsAttrs.getContext());
  }

  ReuseGroupOpGenericAdaptorBase(ReuseGroupOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::tlx::ReuseGroupKindAttr getGroupKindAttr() {
    auto attr = ::llvm::cast<::mlir::triton::tlx::ReuseGroupKindAttr>(getProperties().group_kind);
    return attr;
  }

  ::mlir::triton::tlx::ReuseGroupKind getGroupKind();
  ::mlir::IntegerAttr getGroupSizeAttr();
  uint64_t getGroupSize();
};

} // namespace detail
template <typename RangeT>
class ReuseGroupOpGenericAdaptor : public detail::ReuseGroupOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReuseGroupOpGenericAdaptorBase;
  static constexpr int odsIndex_elements = 0;
public:
  ReuseGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ReuseGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ReuseGroupOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ReuseGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ReuseGroupOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ReuseGroupOpGenericAdaptor(RangeT values, const ReuseGroupOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ReuseGroupOp, typename = std::enable_if_t<std::is_same_v<LateInst, ReuseGroupOp>>>
  ReuseGroupOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getElements() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReuseGroupOpAdaptor : public ReuseGroupOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReuseGroupOpGenericAdaptor::ReuseGroupOpGenericAdaptor;
  ReuseGroupOpAdaptor(ReuseGroupOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ReuseGroupOp : public ::mlir::Op<ReuseGroupOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::triton::tlx::ReuseGroupType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReuseGroupOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReuseGroupOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_elements = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("group_kind"), ::llvm::StringRef("group_size")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getGroupKindAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getGroupKindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getGroupSizeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getGroupSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tlx.reuse_group");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getElements() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getElementsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::tlx::ReuseGroupType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::tlx::ReuseGroupType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::tlx::ReuseGroupKindAttr getGroupKindAttr() {
    return ::llvm::cast<::mlir::triton::tlx::ReuseGroupKindAttr>(getProperties().group_kind);
  }

  ::mlir::triton::tlx::ReuseGroupKind getGroupKind();
  ::mlir::IntegerAttr getGroupSizeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().group_size);
  }

  uint64_t getGroupSize();
  void setGroupKindAttr(::mlir::triton::tlx::ReuseGroupKindAttr attr) {
    getProperties().group_kind = attr;
  }

  void setGroupKind(::mlir::triton::tlx::ReuseGroupKind attrValue);
  void setGroupSizeAttr(::mlir::IntegerAttr attr) {
    getProperties().group_size = attr;
  }

  void setGroupSize(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size = nullptr);
  static ReuseGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size = nullptr);
  static ReuseGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size = nullptr);
  static ReuseGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size = nullptr);
  static ReuseGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size = 1);
  static ReuseGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size = 1);
  static ReuseGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size = 1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size = 1);
  static ReuseGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size = 1);
  static ReuseGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size = 1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReuseGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReuseGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ReuseGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ReuseGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::ReuseGroupOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::detail::ReuseGroupOpGenericAdaptorBase::Properties)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::SetBufferOverlapOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SetBufferOverlapOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  SetBufferOverlapOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tlx.set_buffer_overlap", odsAttrs.getContext());
  }

  SetBufferOverlapOpGenericAdaptorBase(SetBufferOverlapOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class SetBufferOverlapOpGenericAdaptor : public detail::SetBufferOverlapOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SetBufferOverlapOpGenericAdaptorBase;
  static constexpr int odsIndex_storage_alias_spec = 0;
  static constexpr int odsIndex_overlap_def = 1;
public:
  SetBufferOverlapOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SetBufferOverlapOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : SetBufferOverlapOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  SetBufferOverlapOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : SetBufferOverlapOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  SetBufferOverlapOpGenericAdaptor(RangeT values, const SetBufferOverlapOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SetBufferOverlapOp, typename = std::enable_if_t<std::is_same_v<LateInst, SetBufferOverlapOp>>>
  SetBufferOverlapOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getStorageAliasSpec() {
    return (*getODSOperands(0).begin());
  }

  ValueT getOverlapDef() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SetBufferOverlapOpAdaptor : public SetBufferOverlapOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SetBufferOverlapOpGenericAdaptor::SetBufferOverlapOpGenericAdaptor;
  SetBufferOverlapOpAdaptor(SetBufferOverlapOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SetBufferOverlapOp : public ::mlir::Op<SetBufferOverlapOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SetBufferOverlapOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SetBufferOverlapOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_storage_alias_spec = 0;
  static constexpr int odsIndex_overlap_def = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tlx.set_buffer_overlap");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::tlx::StorageAliasSpecType> getStorageAliasSpec() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::tlx::StorageAliasSpecType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::triton::tlx::ReuseGroupType> getOverlapDef() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::tlx::ReuseGroupType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getStorageAliasSpecMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getOverlapDefMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def);
  static SetBufferOverlapOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def);
  static SetBufferOverlapOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def);
  static SetBufferOverlapOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def);
  static SetBufferOverlapOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SetBufferOverlapOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SetBufferOverlapOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static SetBufferOverlapOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static SetBufferOverlapOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::SetBufferOverlapOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::StorageAliasLocalAllocOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class StorageAliasLocalAllocOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  StorageAliasLocalAllocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tlx.storage_alias_local_alloc", odsAttrs.getContext());
  }

  StorageAliasLocalAllocOpGenericAdaptorBase(StorageAliasLocalAllocOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class StorageAliasLocalAllocOpGenericAdaptor : public detail::StorageAliasLocalAllocOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::StorageAliasLocalAllocOpGenericAdaptorBase;
  static constexpr int odsIndex_storage_alias = 0;
public:
  StorageAliasLocalAllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  StorageAliasLocalAllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : StorageAliasLocalAllocOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  StorageAliasLocalAllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : StorageAliasLocalAllocOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  StorageAliasLocalAllocOpGenericAdaptor(RangeT values, const StorageAliasLocalAllocOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = StorageAliasLocalAllocOp, typename = std::enable_if_t<std::is_same_v<LateInst, StorageAliasLocalAllocOp>>>
  StorageAliasLocalAllocOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getStorageAlias() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class StorageAliasLocalAllocOpAdaptor : public StorageAliasLocalAllocOpGenericAdaptor<::mlir::ValueRange> {
public:
  using StorageAliasLocalAllocOpGenericAdaptor::StorageAliasLocalAllocOpGenericAdaptor;
  StorageAliasLocalAllocOpAdaptor(StorageAliasLocalAllocOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class StorageAliasLocalAllocOp : public ::mlir::Op<StorageAliasLocalAllocOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::triton::gpu::MemDescType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StorageAliasLocalAllocOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = StorageAliasLocalAllocOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_storage_alias = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tlx.storage_alias_local_alloc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::tlx::StorageAliasSpecType> getStorageAlias() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::tlx::StorageAliasSpecType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getStorageAliasMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value storage_alias);
  static StorageAliasLocalAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value storage_alias);
  static StorageAliasLocalAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value storage_alias);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias);
  static StorageAliasLocalAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias);
  static StorageAliasLocalAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static StorageAliasLocalAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static StorageAliasLocalAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static StorageAliasLocalAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static StorageAliasLocalAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::StorageAliasLocalAllocOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::StorageAliasSpecOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class StorageAliasSpecOpGenericAdaptorBase {
public:
  struct Properties {
    using buffer_shapeTy = ::mlir::DenseI64ArrayAttr;
    buffer_shapeTy buffer_shape;

    auto getBufferShape() const {
      auto &propStorage = this->buffer_shape;
      return ::llvm::dyn_cast_or_null<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setBufferShape(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->buffer_shape = propValue;
    }
    using buffer_size_bytesTy = ::mlir::IntegerAttr;
    buffer_size_bytesTy buffer_size_bytes;

    auto getBufferSizeBytes() const {
      auto &propStorage = this->buffer_size_bytes;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setBufferSizeBytes(const ::mlir::IntegerAttr &propValue) {
      this->buffer_size_bytes = propValue;
    }
    using storageTy = ::mlir::triton::tlx::StorageKindAttr;
    storageTy storage;

    auto getStorage() const {
      auto &propStorage = this->storage;
      return ::llvm::cast<::mlir::triton::tlx::StorageKindAttr>(propStorage);
    }
    void setStorage(const ::mlir::triton::tlx::StorageKindAttr &propValue) {
      this->storage = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.buffer_shape == this->buffer_shape &&
        rhs.buffer_size_bytes == this->buffer_size_bytes &&
        rhs.storage == this->storage &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  StorageAliasSpecOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tlx.storage_alias_spec", odsAttrs.getContext());
  }

  StorageAliasSpecOpGenericAdaptorBase(StorageAliasSpecOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::tlx::StorageKindAttr getStorageAttr() {
    auto attr = ::llvm::cast<::mlir::triton::tlx::StorageKindAttr>(getProperties().storage);
    return attr;
  }

  ::mlir::triton::tlx::StorageKind getStorage();
  ::mlir::IntegerAttr getBufferSizeBytesAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().buffer_size_bytes);
    return attr;
  }

  ::std::optional<uint64_t> getBufferSizeBytes();
  ::mlir::DenseI64ArrayAttr getBufferShapeAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::DenseI64ArrayAttr>(getProperties().buffer_shape);
    return attr;
  }

  ::std::optional<::llvm::ArrayRef<int64_t>> getBufferShape();
};

} // namespace detail
template <typename RangeT>
class StorageAliasSpecOpGenericAdaptor : public detail::StorageAliasSpecOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::StorageAliasSpecOpGenericAdaptorBase;
public:
  StorageAliasSpecOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  StorageAliasSpecOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : StorageAliasSpecOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  StorageAliasSpecOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : StorageAliasSpecOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  StorageAliasSpecOpGenericAdaptor(RangeT values, const StorageAliasSpecOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = StorageAliasSpecOp, typename = std::enable_if_t<std::is_same_v<LateInst, StorageAliasSpecOp>>>
  StorageAliasSpecOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class StorageAliasSpecOpAdaptor : public StorageAliasSpecOpGenericAdaptor<::mlir::ValueRange> {
public:
  using StorageAliasSpecOpGenericAdaptor::StorageAliasSpecOpGenericAdaptor;
  StorageAliasSpecOpAdaptor(StorageAliasSpecOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class StorageAliasSpecOp : public ::mlir::Op<StorageAliasSpecOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::triton::tlx::StorageAliasSpecType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StorageAliasSpecOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = StorageAliasSpecOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("buffer_shape"), ::llvm::StringRef("buffer_size_bytes"), ::llvm::StringRef("storage")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBufferShapeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBufferShapeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBufferSizeBytesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBufferSizeBytesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStorageAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStorageAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tlx.storage_alias_spec");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::tlx::StorageAliasSpecType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::tlx::StorageAliasSpecType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::tlx::StorageKindAttr getStorageAttr() {
    return ::llvm::cast<::mlir::triton::tlx::StorageKindAttr>(getProperties().storage);
  }

  ::mlir::triton::tlx::StorageKind getStorage();
  ::mlir::IntegerAttr getBufferSizeBytesAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().buffer_size_bytes);
  }

  ::std::optional<uint64_t> getBufferSizeBytes();
  ::mlir::DenseI64ArrayAttr getBufferShapeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DenseI64ArrayAttr>(getProperties().buffer_shape);
  }

  ::std::optional<::llvm::ArrayRef<int64_t>> getBufferShape();
  void setStorageAttr(::mlir::triton::tlx::StorageKindAttr attr) {
    getProperties().storage = attr;
  }

  void setStorage(::mlir::triton::tlx::StorageKind attrValue);
  void setBufferSizeBytesAttr(::mlir::IntegerAttr attr) {
    getProperties().buffer_size_bytes = attr;
  }

  void setBufferSizeBytes(::std::optional<uint64_t> attrValue);
  void setBufferShapeAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().buffer_shape = attr;
  }

  void setBufferShape(::std::optional<::llvm::ArrayRef<int64_t>> attrValue);
  ::mlir::Attribute removeBufferSizeBytesAttr() {
      auto attr = getProperties().buffer_size_bytes;
      getProperties().buffer_size_bytes = {};
      return attr;
  }

  ::mlir::Attribute removeBufferShapeAttr() {
      auto attr = getProperties().buffer_shape;
      getProperties().buffer_shape = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static StorageAliasSpecOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static StorageAliasSpecOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static StorageAliasSpecOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static StorageAliasSpecOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static StorageAliasSpecOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static StorageAliasSpecOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static StorageAliasSpecOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static StorageAliasSpecOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static StorageAliasSpecOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static StorageAliasSpecOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static StorageAliasSpecOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static StorageAliasSpecOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::StorageAliasSpecOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::detail::StorageAliasSpecOpGenericAdaptorBase::Properties)


#endif // GET_OP_CLASSES

