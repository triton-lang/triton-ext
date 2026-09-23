/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::triton::tlx::DummyRegisterLayoutAttr,
::mlir::triton::tlx::DummyTMEMLayoutAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::tlx::DummyRegisterLayoutAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::tlx::DummyRegisterLayoutAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::tlx::DummyTMEMLayoutAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::tlx::DummyTMEMLayoutAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::triton::tlx::DummyRegisterLayoutAttr>([&](auto t) {
      printer << ::mlir::triton::tlx::DummyRegisterLayoutAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::tlx::DummyTMEMLayoutAttr>([&](auto t) {
      printer << ::mlir::triton::tlx::DummyTMEMLayoutAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::triton::tlx {

namespace detail {

struct DummyRegisterLayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type, bool>;
  DummyRegisterLayoutAttrStorage(::llvm::ArrayRef<int64_t> shape, Type elementType, bool tmemCompatible) : shape(std::move(shape)), elementType(std::move(elementType)), tmemCompatible(std::move(tmemCompatible)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, tmemCompatible);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (tmemCompatible == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static DummyRegisterLayoutAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto tmemCompatible = std::move(std::get<2>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<DummyRegisterLayoutAttrStorage>()) DummyRegisterLayoutAttrStorage(std::move(shape), std::move(elementType), std::move(tmemCompatible));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
  bool tmemCompatible;
};

} // namespace detail
DummyRegisterLayoutAttr DummyRegisterLayoutAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType, bool tmemCompatible) {
  return Base::get(context, std::move(shape), std::move(elementType), std::move(tmemCompatible));
}

::mlir::Attribute DummyRegisterLayoutAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_shape;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<bool> _result_tmemCompatible;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'shape'
  _result_shape = ::mlir::FieldParser<::llvm::SmallVector<int64_t>>::parse(odsParser);
  if (::mlir::failed(_result_shape)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TLX_DummyRegisterLayoutAttr parameter 'shape' which is to be a `::llvm::ArrayRef<int64_t>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TLX_DummyRegisterLayoutAttr parameter 'elementType' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'tmemCompatible'
  _result_tmemCompatible = ::mlir::FieldParser<bool>::parse(odsParser);
  if (::mlir::failed(_result_tmemCompatible)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TLX_DummyRegisterLayoutAttr parameter 'tmemCompatible' which is to be a `bool`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_shape));
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_tmemCompatible));
  return DummyRegisterLayoutAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((*_result_shape)),
      Type((*_result_elementType)),
      bool((*_result_tmemCompatible)));
}

void DummyRegisterLayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "[";
  odsPrinter.printStrippedAttrOrType(getShape());
  odsPrinter << "]";
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getTmemCompatible());
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> DummyRegisterLayoutAttr::getShape() const {
  return getImpl()->shape;
}

Type DummyRegisterLayoutAttr::getElementType() const {
  return getImpl()->elementType;
}

bool DummyRegisterLayoutAttr::getTmemCompatible() const {
  return getImpl()->tmemCompatible;
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::DummyRegisterLayoutAttr)
namespace mlir::triton::tlx {

::mlir::Attribute DummyTMEMLayoutAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  return DummyTMEMLayoutAttr::get(odsParser.getContext());
}

void DummyTMEMLayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::DummyTMEMLayoutAttr)
namespace mlir::triton::tlx {


/// Parse an attribute registered to this dialect.
::mlir::Attribute TLXDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void TLXDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::triton::tlx

#endif // GET_ATTRDEF_CLASSES

