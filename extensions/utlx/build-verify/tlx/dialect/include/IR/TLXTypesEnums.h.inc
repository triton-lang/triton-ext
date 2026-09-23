/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TLXTypes.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::tlx {

// TLX reuse group kind for buffer overlap definitions
enum class ReuseGroupKind : uint32_t {
  shared = 0,
  distinct = 1,
};

::std::optional<ReuseGroupKind> symbolizeReuseGroupKind(uint32_t);
::llvm::StringRef stringifyReuseGroupKind(ReuseGroupKind);
::std::optional<ReuseGroupKind> symbolizeReuseGroupKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForReuseGroupKind() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ReuseGroupKind enumValue) {
  return stringifyReuseGroupKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ReuseGroupKind> symbolizeEnum<ReuseGroupKind>(::llvm::StringRef str) {
  return symbolizeReuseGroupKind(str);
}

class ReuseGroupKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ReuseGroupKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ReuseGroupKindAttr get(::mlir::MLIRContext *context, ReuseGroupKind val);
  ReuseGroupKind getValue() const;
};

} // namespace mlir::triton::tlx

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::tlx::ReuseGroupKind, ::mlir::triton::tlx::ReuseGroupKind> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::tlx::ReuseGroupKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for TLX reuse group kind for buffer overlap definitions");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::tlx::ReuseGroupKind> attr = ::mlir::triton::tlx::symbolizeEnum<::mlir::triton::tlx::ReuseGroupKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [shared, distinct] for TLX reuse group kind for buffer overlap definitions, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::tlx::ReuseGroupKind>, std::optional<::mlir::triton::tlx::ReuseGroupKind>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::tlx::ReuseGroupKind>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"shared", "distinct"})))
      return std::optional<::mlir::triton::tlx::ReuseGroupKind>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::tlx::ReuseGroupKind> attr = ::mlir::triton::tlx::symbolizeEnum<::mlir::triton::tlx::ReuseGroupKind>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [shared, distinct] for TLX reuse group kind for buffer overlap definitions, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::tlx::ReuseGroupKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::tlx::ReuseGroupKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::tlx::ReuseGroupKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::tlx::ReuseGroupKind &lhs, const ::mlir::triton::tlx::ReuseGroupKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::tlx {

// TLX storage kind for shared buffers
enum class StorageKind : uint32_t {
  smem = 0,
  tmem = 1,
};

::std::optional<StorageKind> symbolizeStorageKind(uint32_t);
::llvm::StringRef stringifyStorageKind(StorageKind);
::std::optional<StorageKind> symbolizeStorageKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForStorageKind() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(StorageKind enumValue) {
  return stringifyStorageKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<StorageKind> symbolizeEnum<StorageKind>(::llvm::StringRef str) {
  return symbolizeStorageKind(str);
}

class StorageKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = StorageKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static StorageKindAttr get(::mlir::MLIRContext *context, StorageKind val);
  StorageKind getValue() const;
};

} // namespace mlir::triton::tlx

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::tlx::StorageKind, ::mlir::triton::tlx::StorageKind> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::tlx::StorageKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for TLX storage kind for shared buffers");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::tlx::StorageKind> attr = ::mlir::triton::tlx::symbolizeEnum<::mlir::triton::tlx::StorageKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [smem, tmem] for TLX storage kind for shared buffers, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::tlx::StorageKind>, std::optional<::mlir::triton::tlx::StorageKind>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::tlx::StorageKind>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"smem", "tmem"})))
      return std::optional<::mlir::triton::tlx::StorageKind>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::tlx::StorageKind> attr = ::mlir::triton::tlx::symbolizeEnum<::mlir::triton::tlx::StorageKind>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [smem, tmem] for TLX storage kind for shared buffers, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::tlx::StorageKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::tlx::StorageKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static unsigned getHashValue(const ::mlir::triton::tlx::StorageKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::tlx::StorageKind &lhs, const ::mlir::triton::tlx::StorageKind &rhs) {
    return lhs == rhs;
  }
};
}

