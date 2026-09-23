/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::triton::tlx {

/// A storage alias specification type
/// Represents a storage alias specification that can be referenced by multiple
/// local memory allocations. This type carries the storage kind and
/// optional explicit size.
/// 
/// This type is used by the `storage_alias_spec` operation to define a
/// logical grouping for buffer sharing. Multiple `local_alloc` operations
/// can reference the same storage alias specification via the `reuse` parameter.
/// 
/// The actual memory allocation is deferred until `local_alloc` operations
/// reference this storage alias spec. The compiler will:
/// - If `bufferSizeBytes` is specified: verify all references fit within
///   the specified size.
/// - Otherwise: compute the size as the maximum of all referencing allocations.
/// 
/// Note: Only smem and tmem storage kinds are supported. smemCluster is
/// not allowed for storage alias specifications.
/// 
/// Example:
/// ```mlir
/// %alias = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
/// %alias_sized = tlx.storage_alias_spec storage = tmem, size = 16384 : !tlx.storage_alias_spec<tmem, 16384>
/// ```
class StorageAliasSpecType;
/// A reuse group type for buffer overlap definitions
/// Represents a reuse group that defines buffer overlap relationships for
/// shared memory allocations. A reuse group organizes multiple buffers
/// (or nested groups) with a specific relationship type:
/// 
/// - **shared**: Elements logically occupy the same memory region at each
///   buffer index. Useful when buffers are used at different times.
/// - **distinct**: Elements must be in non-overlapping memory regions.
///   Useful when buffers need to be accessed simultaneously.
/// 
/// The reuse group forms a tree structure where leaf nodes are memory
/// allocations and internal nodes are nested reuse groups.
/// 
/// Constraints:
/// - All elements must have the same buffer count (num).
/// - All elements must use the same storage kind (smem or tmem).
///   The storage kind is inferred from the elements and not stored in the type.
/// 
/// Example:
/// ```mlir
/// // A and B share the same memory (used at different times)
/// %group = tlx.reuse_group(%a, %b) {group_type = shared}
///          : (!ttg.memdesc<...>, !ttg.memdesc<...>) -> !tlx.reuse_group<shared, 2>
/// 
/// // Nested groups for complex sharing schemes
/// %inner = tlx.reuse_group(%c, %d, %e) {group_type = distinct}
///          : (...) -> !tlx.reuse_group<distinct, 2>
/// %outer = tlx.reuse_group(%a, %inner) {group_type = shared}
///          : (...) -> !tlx.reuse_group<shared, 2>
/// ```
class ReuseGroupType;
namespace detail {

struct StorageAliasSpecTypeStorage;

} // namespace detail
class StorageAliasSpecType : public ::mlir::Type::TypeBase<StorageAliasSpecType, ::mlir::Type, detail::StorageAliasSpecTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tlx.storage_alias_spec";
  static constexpr ::llvm::StringLiteral dialectName = "tlx";
  using Base::getChecked;
  static StorageAliasSpecType get(::mlir::MLIRContext *context, ::mlir::triton::tlx::StorageKind storage, std::optional<int64_t> bufferSizeBytes);
  static StorageAliasSpecType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::triton::tlx::StorageKind storage, std::optional<int64_t> bufferSizeBytes);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::tlx::StorageKind storage, std::optional<int64_t> bufferSizeBytes);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::tlx::StorageKind storage, std::optional<int64_t> bufferSizeBytes);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"storage_alias_spec"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::tlx::StorageKind getStorage() const;
  std::optional<int64_t> getBufferSizeBytes() const;
};
namespace detail {

struct ReuseGroupTypeStorage;

} // namespace detail
class ReuseGroupType : public ::mlir::Type::TypeBase<ReuseGroupType, ::mlir::Type, detail::ReuseGroupTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tlx.reuse_group";
  static constexpr ::llvm::StringLiteral dialectName = "tlx";
  using Base::getChecked;
  static ReuseGroupType get(::mlir::MLIRContext *context, ::mlir::triton::tlx::ReuseGroupKind groupKind);
  static ReuseGroupType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::triton::tlx::ReuseGroupKind groupKind);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::tlx::ReuseGroupKind groupKind);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::tlx::ReuseGroupKind groupKind);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"reuse_group"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::tlx::ReuseGroupKind getGroupKind() const;
};

} // namespace mlir::triton::tlx
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::StorageAliasSpecType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::ReuseGroupType)

#endif // GET_TYPEDEF_CLASSES

