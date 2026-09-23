/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TLXTypes.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::tlx {

::llvm::StringRef stringifyReuseGroupKind(ReuseGroupKind val) {
  switch (val) {
    case ReuseGroupKind::shared: return "shared";
    case ReuseGroupKind::distinct: return "distinct";
  }
  return "";
}

::std::optional<ReuseGroupKind> symbolizeReuseGroupKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ReuseGroupKind>>(str)
      .Case("shared", ReuseGroupKind::shared)
      .Case("distinct", ReuseGroupKind::distinct)
      .Default(::std::nullopt);
}
::std::optional<ReuseGroupKind> symbolizeReuseGroupKind(uint32_t value) {
  switch (value) {
  case 0: return ReuseGroupKind::shared;
  case 1: return ReuseGroupKind::distinct;
  default: return ::std::nullopt;
  }
}

bool ReuseGroupKindAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
ReuseGroupKindAttr ReuseGroupKindAttr::get(::mlir::MLIRContext *context, ReuseGroupKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<ReuseGroupKindAttr>(baseAttr);
}
ReuseGroupKind ReuseGroupKindAttr::getValue() const {
  return static_cast<ReuseGroupKind>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::triton::tlx
namespace mlir::triton::tlx {

::llvm::StringRef stringifyStorageKind(StorageKind val) {
  switch (val) {
    case StorageKind::smem: return "smem";
    case StorageKind::tmem: return "tmem";
  }
  return "";
}

::std::optional<StorageKind> symbolizeStorageKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<StorageKind>>(str)
      .Case("smem", StorageKind::smem)
      .Case("tmem", StorageKind::tmem)
      .Default(::std::nullopt);
}
::std::optional<StorageKind> symbolizeStorageKind(uint32_t value) {
  switch (value) {
  case 0: return StorageKind::smem;
  case 1: return StorageKind::tmem;
  default: return ::std::nullopt;
  }
}

bool StorageKindAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
StorageKindAttr StorageKindAttr::get(::mlir::MLIRContext *context, StorageKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<StorageKindAttr>(baseAttr);
}
StorageKind StorageKindAttr::getValue() const {
  return static_cast<StorageKind>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::triton::tlx
