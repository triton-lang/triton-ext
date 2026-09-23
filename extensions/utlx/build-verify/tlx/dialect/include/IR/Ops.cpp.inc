/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TLXOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::triton::tlx::LocalAliasOp,
::mlir::triton::tlx::ReleaseLayoutOp,
::mlir::triton::tlx::RequireLayoutOp,
::mlir::triton::tlx::ReuseGroupOp,
::mlir::triton::tlx::SetBufferOverlapOp,
::mlir::triton::tlx::StorageAliasLocalAllocOp,
::mlir::triton::tlx::StorageAliasSpecOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TLXOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::gpu::TensorOrMemDesc>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be TensorOrMemDesc instance, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TLXOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((::llvm::isa<::mlir::Float8E4M3FNType>(elementType))) || ((::llvm::isa<::mlir::Float8E4M3FNUZType>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2Type>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2FNUZType>(elementType))) || ((elementType.isF16())) || ((::llvm::isa<::mlir::BFloat16Type>(elementType))) || ((elementType.isF32())) || ((elementType.isF64()))) || (((elementType.isSignlessInteger(1))) || ((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((::mlir::isa<::mlir::triton::PointerType>(elementType))) && ([](::mlir::Type pointeeType) { return !((::llvm::isa<::mlir::TokenType>(pointeeType))); }(::mlir::cast<::mlir::triton::PointerType>(elementType).getPointeeType()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of floating-point or integer or ptr values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TLXOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::triton::gpu::MemDescType>(type))) || ((::llvm::isa<::mlir::triton::tlx::ReuseGroupType>(type))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of buffered tensor (!ttg.memdesc) or nested reuse group (!tlx.reuse_group), but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TLXOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::tlx::ReuseGroupType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be A reuse group type for buffer overlap definitions, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TLXOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::tlx::StorageAliasSpecType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be A storage alias specification type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TLXOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::gpu::MemDescType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memory descriptor type (`::mlir::triton::gpu::MemDescType`) in Triton IR type system, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TLXOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::tlx::ReuseGroupKindAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: TLX reuse group kind for buffer overlap definitions";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TLXOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TLXOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TLXOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TLXOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TLXOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TLXOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::tlx::StorageKindAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: TLX storage kind for shared buffers";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TLXOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TLXOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TLXOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DenseI64ArrayAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: i64 dense array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TLXOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TLXOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::LocalAliasOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

LocalAliasOpGenericAdaptorBase::LocalAliasOpGenericAdaptorBase(LocalAliasOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
LocalAliasOpAdaptor::LocalAliasOpAdaptor(LocalAliasOp op) : LocalAliasOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LocalAliasOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult LocalAliasOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute LocalAliasOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code LocalAliasOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> LocalAliasOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void LocalAliasOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void LocalAliasOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult LocalAliasOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void LocalAliasOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src) {
  odsState.addOperands(src);
  odsState.addTypes(result);
}

LocalAliasOp LocalAliasOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(src)>(src));
  auto __res__ = ::llvm::dyn_cast<LocalAliasOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LocalAliasOp LocalAliasOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(src)>(src));
}

void LocalAliasOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  odsState.addOperands(src);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

LocalAliasOp LocalAliasOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src));
  auto __res__ = ::llvm::dyn_cast<LocalAliasOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LocalAliasOp LocalAliasOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src));
}

void LocalAliasOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LocalAliasOp LocalAliasOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<LocalAliasOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LocalAliasOp LocalAliasOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void LocalAliasOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LocalAliasOp LocalAliasOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<LocalAliasOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LocalAliasOp LocalAliasOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult LocalAliasOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LocalAliasOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LocalAliasOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::gpu::TensorOrMemDesc type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::triton::gpu::TensorOrMemDesc type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LocalAliasOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::gpu::TensorOrMemDesc>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::gpu::TensorOrMemDesc>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void LocalAliasOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::LocalAliasOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::ReleaseLayoutOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ReleaseLayoutOpGenericAdaptorBase::ReleaseLayoutOpGenericAdaptorBase(ReleaseLayoutOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
ReleaseLayoutOpAdaptor::ReleaseLayoutOpAdaptor(ReleaseLayoutOp op) : ReleaseLayoutOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ReleaseLayoutOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult ReleaseLayoutOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute ReleaseLayoutOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ReleaseLayoutOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> ReleaseLayoutOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void ReleaseLayoutOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void ReleaseLayoutOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult ReleaseLayoutOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void ReleaseLayoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src) {
  odsState.addOperands(src);
  odsState.addTypes(result);
}

ReleaseLayoutOp ReleaseLayoutOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(src)>(src));
  auto __res__ = ::llvm::dyn_cast<ReleaseLayoutOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReleaseLayoutOp ReleaseLayoutOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(src)>(src));
}

void ReleaseLayoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  odsState.addOperands(src);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ReleaseLayoutOp ReleaseLayoutOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src));
  auto __res__ = ::llvm::dyn_cast<ReleaseLayoutOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReleaseLayoutOp ReleaseLayoutOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src));
}

void ReleaseLayoutOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ReleaseLayoutOp ReleaseLayoutOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ReleaseLayoutOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReleaseLayoutOp ReleaseLayoutOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ReleaseLayoutOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ReleaseLayoutOp ReleaseLayoutOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ReleaseLayoutOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReleaseLayoutOp ReleaseLayoutOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ReleaseLayoutOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ReleaseLayoutOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReleaseLayoutOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReleaseLayoutOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::RankedTensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::RankedTensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ReleaseLayoutOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::ReleaseLayoutOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::RequireLayoutOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

RequireLayoutOpGenericAdaptorBase::RequireLayoutOpGenericAdaptorBase(RequireLayoutOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
RequireLayoutOpAdaptor::RequireLayoutOpAdaptor(RequireLayoutOp op) : RequireLayoutOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RequireLayoutOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult RequireLayoutOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute RequireLayoutOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code RequireLayoutOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> RequireLayoutOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void RequireLayoutOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void RequireLayoutOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult RequireLayoutOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void RequireLayoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src) {
  odsState.addOperands(src);
  odsState.addTypes(result);
}

RequireLayoutOp RequireLayoutOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(src)>(src));
  auto __res__ = ::llvm::dyn_cast<RequireLayoutOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RequireLayoutOp RequireLayoutOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(src)>(src));
}

void RequireLayoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  odsState.addOperands(src);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

RequireLayoutOp RequireLayoutOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src));
  auto __res__ = ::llvm::dyn_cast<RequireLayoutOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RequireLayoutOp RequireLayoutOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src));
}

void RequireLayoutOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RequireLayoutOp RequireLayoutOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<RequireLayoutOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RequireLayoutOp RequireLayoutOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void RequireLayoutOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RequireLayoutOp RequireLayoutOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<RequireLayoutOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RequireLayoutOp RequireLayoutOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult RequireLayoutOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult RequireLayoutOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult RequireLayoutOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::gpu::TensorOrMemDesc type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::triton::gpu::TensorOrMemDesc type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RequireLayoutOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::gpu::TensorOrMemDesc>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::gpu::TensorOrMemDesc>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void RequireLayoutOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::RequireLayoutOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::ReuseGroupOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ReuseGroupOpGenericAdaptorBase::ReuseGroupOpGenericAdaptorBase(ReuseGroupOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ReuseGroupOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::triton::tlx::ReuseGroupKind ReuseGroupOpGenericAdaptorBase::getGroupKind() {
  auto attr = getGroupKindAttr();
  return attr.getValue();
}

::mlir::IntegerAttr ReuseGroupOpGenericAdaptorBase::getGroupSizeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().group_size);
  return attr;
}

uint64_t ReuseGroupOpGenericAdaptorBase::getGroupSize() {
  auto attr = getGroupSizeAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
ReuseGroupOpAdaptor::ReuseGroupOpAdaptor(ReuseGroupOp op) : ReuseGroupOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ReuseGroupOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_group_kind = getProperties().group_kind; (void)tblgen_group_kind;
  if (!tblgen_group_kind) return emitError(loc, "'tlx.reuse_group' op requires attribute 'group_kind'");
  auto tblgen_group_size = getProperties().group_size; (void)tblgen_group_size;

  if (tblgen_group_kind && !((::llvm::isa<::mlir::triton::tlx::ReuseGroupKindAttr>(tblgen_group_kind))))
    return emitError(loc, "'tlx.reuse_group' op attribute 'group_kind' failed to satisfy constraint: TLX reuse group kind for buffer overlap definitions");

  if (tblgen_group_size && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_group_size))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_group_size).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tlx.reuse_group' op attribute 'group_size' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReuseGroupOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange ReuseGroupOp::getElementsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult ReuseGroupOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.group_kind;
       auto attr = dict.get("group_kind");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `group_kind` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.group_size;
       auto attr = dict.get("group_size");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `group_size` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReuseGroupOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.group_kind;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("group_kind",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.group_size;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("group_size",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ReuseGroupOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.group_kind.getAsOpaquePointer()), 
    llvm::hash_value(prop.group_size.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReuseGroupOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "group_kind")
      return prop.group_kind;

    if (name == "group_size")
      return prop.group_size;
  return std::nullopt;
}

void ReuseGroupOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "group_kind") {
       prop.group_kind = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.group_kind)>>(value);
       return;
    }

    if (name == "group_size") {
       prop.group_size = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.group_size)>>(value);
       return;
    }
}

void ReuseGroupOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.group_kind) attrs.append("group_kind", prop.group_kind);

    if (prop.group_size) attrs.append("group_size", prop.group_size);
}

::llvm::LogicalResult ReuseGroupOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getGroupKindAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps1(attr, "group_kind", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getGroupSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps2(attr, "group_size", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ReuseGroupOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.group_kind)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.group_size)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReuseGroupOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.group_kind);

  writer.writeOptionalAttribute(prop.group_size);
}

::mlir::triton::tlx::ReuseGroupKind ReuseGroupOp::getGroupKind() {
  auto attr = getGroupKindAttr();
  return attr.getValue();
}

uint64_t ReuseGroupOp::getGroupSize() {
  auto attr = getGroupSizeAttr();
  return attr.getValue().getZExtValue();
}

void ReuseGroupOp::setGroupKind(::mlir::triton::tlx::ReuseGroupKind attrValue) {
  getProperties().group_kind = ::mlir::triton::tlx::ReuseGroupKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void ReuseGroupOp::setGroupSize(uint64_t attrValue) {
  getProperties().group_size = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ReuseGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size) {
  odsState.addOperands(elements);
  odsState.getOrAddProperties<Properties>().group_kind = group_kind;
  if (group_size) {
    odsState.getOrAddProperties<Properties>().group_size = group_size;
  }
  odsState.addTypes(result);
}

ReuseGroupOp ReuseGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(elements)>(elements), std::forward<decltype(group_kind)>(group_kind), std::forward<decltype(group_size)>(group_size));
  auto __res__ = ::llvm::dyn_cast<ReuseGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReuseGroupOp ReuseGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(elements)>(elements), std::forward<decltype(group_kind)>(group_kind), std::forward<decltype(group_size)>(group_size));
}

void ReuseGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size) {
  odsState.addOperands(elements);
  odsState.getOrAddProperties<Properties>().group_kind = group_kind;
  if (group_size) {
    odsState.getOrAddProperties<Properties>().group_size = group_size;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ReuseGroupOp ReuseGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(elements)>(elements), std::forward<decltype(group_kind)>(group_kind), std::forward<decltype(group_size)>(group_size));
  auto __res__ = ::llvm::dyn_cast<ReuseGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReuseGroupOp ReuseGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKindAttr group_kind, ::mlir::IntegerAttr group_size) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(elements)>(elements), std::forward<decltype(group_kind)>(group_kind), std::forward<decltype(group_size)>(group_size));
}

void ReuseGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size) {
  odsState.addOperands(elements);
  odsState.getOrAddProperties<Properties>().group_kind = ::mlir::triton::tlx::ReuseGroupKindAttr::get(odsBuilder.getContext(), group_kind);
  odsState.getOrAddProperties<Properties>().group_size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), group_size);
  odsState.addTypes(result);
}

ReuseGroupOp ReuseGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(elements)>(elements), std::forward<decltype(group_kind)>(group_kind), std::forward<decltype(group_size)>(group_size));
  auto __res__ = ::llvm::dyn_cast<ReuseGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReuseGroupOp ReuseGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(elements)>(elements), std::forward<decltype(group_kind)>(group_kind), std::forward<decltype(group_size)>(group_size));
}

void ReuseGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size) {
  odsState.addOperands(elements);
  odsState.getOrAddProperties<Properties>().group_kind = ::mlir::triton::tlx::ReuseGroupKindAttr::get(odsBuilder.getContext(), group_kind);
  odsState.getOrAddProperties<Properties>().group_size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), group_size);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ReuseGroupOp ReuseGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(elements)>(elements), std::forward<decltype(group_kind)>(group_kind), std::forward<decltype(group_size)>(group_size));
  auto __res__ = ::llvm::dyn_cast<ReuseGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReuseGroupOp ReuseGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange elements, ::mlir::triton::tlx::ReuseGroupKind group_kind, uint64_t group_size) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(elements)>(elements), std::forward<decltype(group_kind)>(group_kind), std::forward<decltype(group_size)>(group_size));
}

void ReuseGroupOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    (void)odsState.getOrAddProperties<ReuseGroupOp::Properties>();
    ::mlir::PropertyRef properties = odsState.getRawProperties();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

ReuseGroupOp ReuseGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ReuseGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReuseGroupOp ReuseGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ReuseGroupOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ReuseGroupOp ReuseGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ReuseGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ReuseGroupOp ReuseGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

void ReuseGroupOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.group_size)
    properties.group_size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1);
}

::llvm::LogicalResult ReuseGroupOp::verifyInvariantsImpl() {
  auto tblgen_group_kind = getProperties().group_kind; (void)tblgen_group_kind;
  if (!tblgen_group_kind) return emitOpError("requires attribute 'group_kind'");
  auto tblgen_group_size = getProperties().group_size; (void)tblgen_group_size;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps1(*this, tblgen_group_kind, "group_kind")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps2(*this, tblgen_group_size, "group_size")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ReuseGroupOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReuseGroupOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> elementsOperands;
  ::llvm::SMLoc elementsOperandsLoc;
  (void)elementsOperandsLoc;
  ::mlir::triton::tlx::ReuseGroupKindAttr group_kindAttr;
  ::mlir::IntegerAttr group_sizeAttr;
  ::llvm::SmallVector<::mlir::Type, 1> elementsTypes;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);
  if (parser.parseLParen())
    return ::mlir::failure();

  elementsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(elementsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseKeyword("group_kind"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"shared","distinct"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "group_kind", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'group_kind' [shared, distinct]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::tlx::symbolizeReuseGroupKind(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "group_kind attribute specification: \"" << attrStr << '"';;

      group_kindAttr = ::mlir::triton::tlx::ReuseGroupKindAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<ReuseGroupOp::Properties>().group_kind = group_kindAttr;
    }
  }
  if (::mlir::succeeded(parser.parseOptionalComma())) {
  if (parser.parseKeyword("group_size"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(group_sizeAttr, parser.getBuilder().getIntegerType(64))) {
    return ::mlir::failure();
  }
  if (group_sizeAttr) result.getOrAddProperties<ReuseGroupOp::Properties>().group_size = group_sizeAttr;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseTypeList(elementsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resultRawType))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(elementsOperands, elementsTypes, elementsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReuseGroupOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getElements();
  _odsPrinter << ")";
  _odsPrinter << ' ' << "group_kind";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = getGroupKind();
    auto caseValueStr = stringifyReuseGroupKind(caseValue);
    _odsPrinter << caseValueStr;
  }
  if (getGroupSizeAttr() != ::mlir::OpBuilder((*this)->getContext()).getIntegerAttr(::mlir::OpBuilder((*this)->getContext()).getIntegerType(64), 1)) {
    _odsPrinter << ",";
    _odsPrinter << ' ' << "group_size";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getGroupSizeAttr());
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("group_kind");
  elidedAttrs.push_back("group_size");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getGroupSizeAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("group_size");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  _odsPrinter << getElements().getTypes();
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
   _odsPrinter << getResult().getType();
}

void ReuseGroupOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::ReuseGroupOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::detail::ReuseGroupOpGenericAdaptorBase::Properties)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::SetBufferOverlapOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

SetBufferOverlapOpGenericAdaptorBase::SetBufferOverlapOpGenericAdaptorBase(SetBufferOverlapOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
SetBufferOverlapOpAdaptor::SetBufferOverlapOpAdaptor(SetBufferOverlapOp op) : SetBufferOverlapOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SetBufferOverlapOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult SetBufferOverlapOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute SetBufferOverlapOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SetBufferOverlapOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> SetBufferOverlapOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void SetBufferOverlapOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void SetBufferOverlapOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult SetBufferOverlapOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void SetBufferOverlapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def) {
  odsState.addOperands(storage_alias_spec);
  odsState.addOperands(overlap_def);
}

SetBufferOverlapOp SetBufferOverlapOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(storage_alias_spec)>(storage_alias_spec), std::forward<decltype(overlap_def)>(overlap_def));
  auto __res__ = ::llvm::dyn_cast<SetBufferOverlapOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

SetBufferOverlapOp SetBufferOverlapOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def) {
  return create(builder, builder.getLoc(), std::forward<decltype(storage_alias_spec)>(storage_alias_spec), std::forward<decltype(overlap_def)>(overlap_def));
}

void SetBufferOverlapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def) {
  odsState.addOperands(storage_alias_spec);
  odsState.addOperands(overlap_def);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

SetBufferOverlapOp SetBufferOverlapOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(storage_alias_spec)>(storage_alias_spec), std::forward<decltype(overlap_def)>(overlap_def));
  auto __res__ = ::llvm::dyn_cast<SetBufferOverlapOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

SetBufferOverlapOp SetBufferOverlapOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias_spec, ::mlir::Value overlap_def) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(storage_alias_spec)>(storage_alias_spec), std::forward<decltype(overlap_def)>(overlap_def));
}

void SetBufferOverlapOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

SetBufferOverlapOp SetBufferOverlapOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<SetBufferOverlapOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

SetBufferOverlapOp SetBufferOverlapOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void SetBufferOverlapOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

SetBufferOverlapOp SetBufferOverlapOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<SetBufferOverlapOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

SetBufferOverlapOp SetBufferOverlapOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult SetBufferOverlapOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult SetBufferOverlapOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SetBufferOverlapOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand storage_alias_specRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> storage_alias_specOperands(&storage_alias_specRawOperand, 1);  ::llvm::SMLoc storage_alias_specOperandsLoc;
  (void)storage_alias_specOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand overlap_defRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> overlap_defOperands(&overlap_defRawOperand, 1);  ::llvm::SMLoc overlap_defOperandsLoc;
  (void)overlap_defOperandsLoc;
  ::mlir::Type storage_alias_specRawType{};
  ::llvm::ArrayRef<::mlir::Type> storage_alias_specTypes(&storage_alias_specRawType, 1);
  ::mlir::Type overlap_defRawType{};
  ::llvm::ArrayRef<::mlir::Type> overlap_defTypes(&overlap_defRawType, 1);
  if (parser.parseLParen())
    return ::mlir::failure();

  storage_alias_specOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(storage_alias_specRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  overlap_defOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(overlap_defRawOperand))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(storage_alias_specRawType))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(overlap_defRawType))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(storage_alias_specOperands, storage_alias_specTypes, storage_alias_specOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(overlap_defOperands, overlap_defTypes, overlap_defOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SetBufferOverlapOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getStorageAliasSpec();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getOverlapDef();
  _odsPrinter << ")";
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
   _odsPrinter << getStorageAliasSpec().getType();
  _odsPrinter << ",";
  _odsPrinter << ' ';
   _odsPrinter << getOverlapDef().getType();
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ' << "(";
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::SetBufferOverlapOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::StorageAliasLocalAllocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

StorageAliasLocalAllocOpGenericAdaptorBase::StorageAliasLocalAllocOpGenericAdaptorBase(StorageAliasLocalAllocOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
StorageAliasLocalAllocOpAdaptor::StorageAliasLocalAllocOpAdaptor(StorageAliasLocalAllocOp op) : StorageAliasLocalAllocOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult StorageAliasLocalAllocOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult StorageAliasLocalAllocOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute StorageAliasLocalAllocOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StorageAliasLocalAllocOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> StorageAliasLocalAllocOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void StorageAliasLocalAllocOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void StorageAliasLocalAllocOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult StorageAliasLocalAllocOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void StorageAliasLocalAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value storage_alias) {
  odsState.addOperands(storage_alias);
  odsState.addTypes(result);
}

StorageAliasLocalAllocOp StorageAliasLocalAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value storage_alias) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(storage_alias)>(storage_alias));
  auto __res__ = ::llvm::dyn_cast<StorageAliasLocalAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasLocalAllocOp StorageAliasLocalAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value storage_alias) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(storage_alias)>(storage_alias));
}

void StorageAliasLocalAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias) {
  odsState.addOperands(storage_alias);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

StorageAliasLocalAllocOp StorageAliasLocalAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(storage_alias)>(storage_alias));
  auto __res__ = ::llvm::dyn_cast<StorageAliasLocalAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasLocalAllocOp StorageAliasLocalAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value storage_alias) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(storage_alias)>(storage_alias));
}

void StorageAliasLocalAllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

StorageAliasLocalAllocOp StorageAliasLocalAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<StorageAliasLocalAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasLocalAllocOp StorageAliasLocalAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void StorageAliasLocalAllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

StorageAliasLocalAllocOp StorageAliasLocalAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<StorageAliasLocalAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasLocalAllocOp StorageAliasLocalAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult StorageAliasLocalAllocOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult StorageAliasLocalAllocOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StorageAliasLocalAllocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand storage_aliasRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> storage_aliasOperands(&storage_aliasRawOperand, 1);  ::llvm::SMLoc storage_aliasOperandsLoc;
  (void)storage_aliasOperandsLoc;
  ::mlir::Type storage_aliasRawType{};
  ::llvm::ArrayRef<::mlir::Type> storage_aliasTypes(&storage_aliasRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  storage_aliasOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(storage_aliasRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(storage_aliasRawType))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resultRawType))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(storage_aliasOperands, storage_aliasTypes, storage_aliasOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StorageAliasLocalAllocOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getStorageAlias();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getStorageAlias().getType();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
   _odsPrinter << getResult().getType();
}

void StorageAliasLocalAllocOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::StorageAliasLocalAllocOp)

namespace mlir::triton::tlx {


//===----------------------------------------------------------------------===//
// ::mlir::triton::tlx::StorageAliasSpecOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

StorageAliasSpecOpGenericAdaptorBase::StorageAliasSpecOpGenericAdaptorBase(StorageAliasSpecOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::triton::tlx::StorageKind StorageAliasSpecOpGenericAdaptorBase::getStorage() {
  auto attr = getStorageAttr();
  return attr.getValue();
}

::std::optional<uint64_t> StorageAliasSpecOpGenericAdaptorBase::getBufferSizeBytes() {
  auto attr = getBufferSizeBytesAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::std::optional<::llvm::ArrayRef<int64_t>> StorageAliasSpecOpGenericAdaptorBase::getBufferShape() {
  auto attr = getBufferShapeAttr();
  return attr ? ::std::optional<::llvm::ArrayRef<int64_t>>(attr) : (::std::nullopt);
}


} // namespace detail
StorageAliasSpecOpAdaptor::StorageAliasSpecOpAdaptor(StorageAliasSpecOp op) : StorageAliasSpecOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult StorageAliasSpecOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_buffer_shape = getProperties().buffer_shape; (void)tblgen_buffer_shape;
  auto tblgen_buffer_size_bytes = getProperties().buffer_size_bytes; (void)tblgen_buffer_size_bytes;
  auto tblgen_storage = getProperties().storage; (void)tblgen_storage;
  if (!tblgen_storage) return emitError(loc, "'tlx.storage_alias_spec' op requires attribute 'storage'");

  if (tblgen_storage && !((::llvm::isa<::mlir::triton::tlx::StorageKindAttr>(tblgen_storage))))
    return emitError(loc, "'tlx.storage_alias_spec' op attribute 'storage' failed to satisfy constraint: TLX storage kind for shared buffers");

  if (tblgen_buffer_size_bytes && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_buffer_size_bytes))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_buffer_size_bytes).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tlx.storage_alias_spec' op attribute 'buffer_size_bytes' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_buffer_shape && !((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_buffer_shape))))
    return emitError(loc, "'tlx.storage_alias_spec' op attribute 'buffer_shape' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

::llvm::LogicalResult StorageAliasSpecOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.buffer_shape;
       auto attr = dict.get("buffer_shape");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `buffer_shape` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.buffer_size_bytes;
       auto attr = dict.get("buffer_size_bytes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `buffer_size_bytes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.storage;
       auto attr = dict.get("storage");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `storage` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StorageAliasSpecOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.buffer_shape;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("buffer_shape",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.buffer_size_bytes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("buffer_size_bytes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.storage;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("storage",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StorageAliasSpecOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.buffer_shape.getAsOpaquePointer()), 
    llvm::hash_value(prop.buffer_size_bytes.getAsOpaquePointer()), 
    llvm::hash_value(prop.storage.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StorageAliasSpecOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "buffer_shape")
      return prop.buffer_shape;

    if (name == "buffer_size_bytes")
      return prop.buffer_size_bytes;

    if (name == "storage")
      return prop.storage;
  return std::nullopt;
}

void StorageAliasSpecOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "buffer_shape") {
       prop.buffer_shape = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.buffer_shape)>>(value);
       return;
    }

    if (name == "buffer_size_bytes") {
       prop.buffer_size_bytes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.buffer_size_bytes)>>(value);
       return;
    }

    if (name == "storage") {
       prop.storage = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.storage)>>(value);
       return;
    }
}

void StorageAliasSpecOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.buffer_shape) attrs.append("buffer_shape", prop.buffer_shape);

    if (prop.buffer_size_bytes) attrs.append("buffer_size_bytes", prop.buffer_size_bytes);

    if (prop.storage) attrs.append("storage", prop.storage);
}

::llvm::LogicalResult StorageAliasSpecOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBufferShapeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps4(attr, "buffer_shape", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBufferSizeBytesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps2(attr, "buffer_size_bytes", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStorageAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps3(attr, "storage", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult StorageAliasSpecOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.buffer_shape)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.buffer_size_bytes)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.storage)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StorageAliasSpecOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.buffer_shape);

  writer.writeOptionalAttribute(prop.buffer_size_bytes);
  writer.writeAttribute(prop.storage);
}

::mlir::triton::tlx::StorageKind StorageAliasSpecOp::getStorage() {
  auto attr = getStorageAttr();
  return attr.getValue();
}

::std::optional<uint64_t> StorageAliasSpecOp::getBufferSizeBytes() {
  auto attr = getBufferSizeBytesAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::std::optional<::llvm::ArrayRef<int64_t>> StorageAliasSpecOp::getBufferShape() {
  auto attr = getBufferShapeAttr();
  return attr ? ::std::optional<::llvm::ArrayRef<int64_t>>(attr) : (::std::nullopt);
}

void StorageAliasSpecOp::setStorage(::mlir::triton::tlx::StorageKind attrValue) {
  getProperties().storage = ::mlir::triton::tlx::StorageKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void StorageAliasSpecOp::setBufferSizeBytes(::std::optional<uint64_t> attrValue) {
    auto &odsProp = getProperties().buffer_size_bytes;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue);
    else
      odsProp = nullptr;
}

void StorageAliasSpecOp::setBufferShape(::std::optional<::llvm::ArrayRef<int64_t>> attrValue) {
    auto &odsProp = getProperties().buffer_shape;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(*attrValue);
    else
      odsProp = nullptr;
}

void StorageAliasSpecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  odsState.getOrAddProperties<Properties>().storage = storage;
  if (buffer_size_bytes) {
    odsState.getOrAddProperties<Properties>().buffer_size_bytes = buffer_size_bytes;
  }
  if (buffer_shape) {
    odsState.getOrAddProperties<Properties>().buffer_shape = buffer_shape;
  }
  odsState.addTypes(result);
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(storage)>(storage), std::forward<decltype(buffer_size_bytes)>(buffer_size_bytes), std::forward<decltype(buffer_shape)>(buffer_shape));
  auto __res__ = ::llvm::dyn_cast<StorageAliasSpecOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(storage)>(storage), std::forward<decltype(buffer_size_bytes)>(buffer_size_bytes), std::forward<decltype(buffer_shape)>(buffer_shape));
}

void StorageAliasSpecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  odsState.getOrAddProperties<Properties>().storage = storage;
  if (buffer_size_bytes) {
    odsState.getOrAddProperties<Properties>().buffer_size_bytes = buffer_size_bytes;
  }
  if (buffer_shape) {
    odsState.getOrAddProperties<Properties>().buffer_shape = buffer_shape;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(storage)>(storage), std::forward<decltype(buffer_size_bytes)>(buffer_size_bytes), std::forward<decltype(buffer_shape)>(buffer_shape));
  auto __res__ = ::llvm::dyn_cast<StorageAliasSpecOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKindAttr storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(storage)>(storage), std::forward<decltype(buffer_size_bytes)>(buffer_size_bytes), std::forward<decltype(buffer_shape)>(buffer_shape));
}

void StorageAliasSpecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  odsState.getOrAddProperties<Properties>().storage = ::mlir::triton::tlx::StorageKindAttr::get(odsBuilder.getContext(), storage);
  if (buffer_size_bytes) {
    odsState.getOrAddProperties<Properties>().buffer_size_bytes = buffer_size_bytes;
  }
  if (buffer_shape) {
    odsState.getOrAddProperties<Properties>().buffer_shape = buffer_shape;
  }
  odsState.addTypes(result);
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(storage)>(storage), std::forward<decltype(buffer_size_bytes)>(buffer_size_bytes), std::forward<decltype(buffer_shape)>(buffer_shape));
  auto __res__ = ::llvm::dyn_cast<StorageAliasSpecOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(storage)>(storage), std::forward<decltype(buffer_size_bytes)>(buffer_size_bytes), std::forward<decltype(buffer_shape)>(buffer_shape));
}

void StorageAliasSpecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  odsState.getOrAddProperties<Properties>().storage = ::mlir::triton::tlx::StorageKindAttr::get(odsBuilder.getContext(), storage);
  if (buffer_size_bytes) {
    odsState.getOrAddProperties<Properties>().buffer_size_bytes = buffer_size_bytes;
  }
  if (buffer_shape) {
    odsState.getOrAddProperties<Properties>().buffer_shape = buffer_shape;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(storage)>(storage), std::forward<decltype(buffer_size_bytes)>(buffer_size_bytes), std::forward<decltype(buffer_shape)>(buffer_shape));
  auto __res__ = ::llvm::dyn_cast<StorageAliasSpecOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::triton::tlx::StorageKind storage, /*optional*/::mlir::IntegerAttr buffer_size_bytes, /*optional*/::mlir::DenseI64ArrayAttr buffer_shape) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(storage)>(storage), std::forward<decltype(buffer_size_bytes)>(buffer_size_bytes), std::forward<decltype(buffer_shape)>(buffer_shape));
}

void StorageAliasSpecOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    (void)odsState.getOrAddProperties<StorageAliasSpecOp::Properties>();
    ::mlir::PropertyRef properties = odsState.getRawProperties();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<StorageAliasSpecOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void StorageAliasSpecOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<StorageAliasSpecOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StorageAliasSpecOp StorageAliasSpecOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult StorageAliasSpecOp::verifyInvariantsImpl() {
  auto tblgen_buffer_shape = getProperties().buffer_shape; (void)tblgen_buffer_shape;
  auto tblgen_buffer_size_bytes = getProperties().buffer_size_bytes; (void)tblgen_buffer_size_bytes;
  auto tblgen_storage = getProperties().storage; (void)tblgen_storage;
  if (!tblgen_storage) return emitOpError("requires attribute 'storage'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps3(*this, tblgen_storage, "storage")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps2(*this, tblgen_buffer_size_bytes, "buffer_size_bytes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TLXOps4(*this, tblgen_buffer_shape, "buffer_shape")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TLXOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult StorageAliasSpecOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StorageAliasSpecOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::triton::tlx::StorageKindAttr storageAttr;
  ::mlir::IntegerAttr buffer_size_bytesAttr;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);
  if (parser.parseKeyword("storage"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"smem","tmem"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "storage", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'storage' [smem, tmem]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::tlx::symbolizeStorageKind(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "storage attribute specification: \"" << attrStr << '"';;

      storageAttr = ::mlir::triton::tlx::StorageKindAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<StorageAliasSpecOp::Properties>().storage = storageAttr;
    }
  }
  if (::mlir::succeeded(parser.parseOptionalComma())) {
  if (parser.parseKeyword("size"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(buffer_size_bytesAttr, parser.getBuilder().getIntegerType(64))) {
    return ::mlir::failure();
  }
  if (buffer_size_bytesAttr) result.getOrAddProperties<StorageAliasSpecOp::Properties>().buffer_size_bytes = buffer_size_bytesAttr;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawType))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  return ::mlir::success();
}

void StorageAliasSpecOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "storage";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = getStorage();
    auto caseValueStr = stringifyStorageKind(caseValue);
    _odsPrinter << caseValueStr;
  }
  if (getBufferSizeBytesAttr()) {
    _odsPrinter << ",";
    _odsPrinter << ' ' << "size";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getBufferSizeBytesAttr());
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("storage");
  elidedAttrs.push_back("buffer_size_bytes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getResult().getType();
}

void StorageAliasSpecOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::tlx
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::StorageAliasSpecOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::tlx::detail::StorageAliasSpecOpGenericAdaptorBase::Properties)


#endif // GET_OP_CLASSES

