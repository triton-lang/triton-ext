/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::triton::tlx::StorageAliasSpecType,
::mlir::triton::tlx::ReuseGroupType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::tlx::StorageAliasSpecType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::tlx::StorageAliasSpecType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::tlx::ReuseGroupType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::tlx::ReuseGroupType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::triton::tlx::StorageAliasSpecType>([&](auto t) {
      printer << ::mlir::triton::tlx::StorageAliasSpecType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::tlx::ReuseGroupType>([&](auto t) {
      printer << ::mlir::triton::tlx::ReuseGroupType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::triton::tlx {

namespace detail {

struct StorageAliasSpecTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::triton::tlx::StorageKind, std::optional<int64_t>>;
  StorageAliasSpecTypeStorage(::mlir::triton::tlx::StorageKind storage, std::optional<int64_t> bufferSizeBytes) : storage(std::move(storage)), bufferSizeBytes(std::move(bufferSizeBytes)) {}

  KeyTy getAsKey() const {
    return KeyTy(storage, bufferSizeBytes);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (storage == std::get<0>(tblgenKey)) && (bufferSizeBytes == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static StorageAliasSpecTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto storage = std::move(std::get<0>(tblgenKey));
    auto bufferSizeBytes = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<StorageAliasSpecTypeStorage>()) StorageAliasSpecTypeStorage(std::move(storage), std::move(bufferSizeBytes));
  }

  ::mlir::triton::tlx::StorageKind storage;
  std::optional<int64_t> bufferSizeBytes;
};

} // namespace detail
StorageAliasSpecType StorageAliasSpecType::get(::mlir::MLIRContext *context, ::mlir::triton::tlx::StorageKind storage, std::optional<int64_t> bufferSizeBytes) {
  return Base::get(context, std::move(storage), std::move(bufferSizeBytes));
}

StorageAliasSpecType StorageAliasSpecType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::triton::tlx::StorageKind storage, std::optional<int64_t> bufferSizeBytes) {
  return Base::getChecked(emitError, context, std::move(storage), std::move(bufferSizeBytes));
}

::llvm::LogicalResult StorageAliasSpecType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::tlx::StorageKind storage, std::optional<int64_t> bufferSizeBytes) {
  if (::mlir::failed(verify(emitError, storage, bufferSizeBytes)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type StorageAliasSpecType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::triton::tlx::StorageKind> _result_storage;
  ::mlir::FailureOr<std::optional<int64_t>> _result_bufferSizeBytes;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'storage'
  _result_storage = [&]() -> ::mlir::FailureOr<::mlir::triton::tlx::StorageKind> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::triton::tlx::symbolizeStorageKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::triton::tlx::StorageKind" << " to be one of: " << "smem" << ", " << "tmem")};
    }();
  if (::mlir::failed(_result_storage)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TLX_StorageAliasSpecType parameter 'storage' which is to be a `::mlir::triton::tlx::StorageKind`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {

    // Parse variable 'bufferSizeBytes'
    _result_bufferSizeBytes = ::mlir::FieldParser<std::optional<int64_t>>::parse(odsParser);
    if (::mlir::failed(_result_bufferSizeBytes)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TLX_StorageAliasSpecType parameter 'bufferSizeBytes' which is to be a `std::optional<int64_t>`");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_storage));
  return odsParser.getChecked<StorageAliasSpecType>(odsLoc, odsParser.getContext(),
      ::mlir::triton::tlx::StorageKind((*_result_storage)),
      std::optional<int64_t>((_result_bufferSizeBytes.value_or(std::optional<int64_t>()))));
}

void StorageAliasSpecType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyStorageKind(getStorage());
  if (!(getBufferSizeBytes() == std::optional<int64_t>())) {
    odsPrinter << ",";
    if (!(getBufferSizeBytes() == std::optional<int64_t>())) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getBufferSizeBytes());
    }
  } else {
  }
  odsPrinter << ">";
}

::mlir::triton::tlx::StorageKind StorageAliasSpecType::getStorage() const {
  return getImpl()->storage;
}

std::optional<int64_t> StorageAliasSpecType::getBufferSizeBytes() const {
  return getImpl()->bufferSizeBytes;
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::StorageAliasSpecType)
namespace mlir::triton::tlx {

namespace detail {

struct ReuseGroupTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::triton::tlx::ReuseGroupKind>;
  ReuseGroupTypeStorage(::mlir::triton::tlx::ReuseGroupKind groupKind) : groupKind(std::move(groupKind)) {}

  KeyTy getAsKey() const {
    return KeyTy(groupKind);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (groupKind == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ReuseGroupTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto groupKind = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ReuseGroupTypeStorage>()) ReuseGroupTypeStorage(std::move(groupKind));
  }

  ::mlir::triton::tlx::ReuseGroupKind groupKind;
};

} // namespace detail
ReuseGroupType ReuseGroupType::get(::mlir::MLIRContext *context, ::mlir::triton::tlx::ReuseGroupKind groupKind) {
  return Base::get(context, std::move(groupKind));
}

ReuseGroupType ReuseGroupType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::triton::tlx::ReuseGroupKind groupKind) {
  return Base::getChecked(emitError, context, std::move(groupKind));
}

::llvm::LogicalResult ReuseGroupType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::triton::tlx::ReuseGroupKind groupKind) {
  if (::mlir::failed(verify(emitError, groupKind)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type ReuseGroupType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::triton::tlx::ReuseGroupKind> _result_groupKind;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'groupKind'
  _result_groupKind = [&]() -> ::mlir::FailureOr<::mlir::triton::tlx::ReuseGroupKind> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::triton::tlx::symbolizeReuseGroupKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::triton::tlx::ReuseGroupKind" << " to be one of: " << "shared" << ", " << "distinct")};
    }();
  if (::mlir::failed(_result_groupKind)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TLX_ReuseGroupType parameter 'groupKind' which is to be a `::mlir::triton::tlx::ReuseGroupKind`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_groupKind));
  return odsParser.getChecked<ReuseGroupType>(odsLoc, odsParser.getContext(),
      ::mlir::triton::tlx::ReuseGroupKind((*_result_groupKind)));
}

void ReuseGroupType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyReuseGroupKind(getGroupKind());
  odsPrinter << ">";
}

::mlir::triton::tlx::ReuseGroupKind ReuseGroupType::getGroupKind() const {
  return getImpl()->groupKind;
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::ReuseGroupType)
namespace mlir::triton::tlx {


/// Parse a type registered to this dialect.
::mlir::Type TLXDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void TLXDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}

} // namespace mlir::triton::tlx

#endif // GET_TYPEDEF_CLASSES

