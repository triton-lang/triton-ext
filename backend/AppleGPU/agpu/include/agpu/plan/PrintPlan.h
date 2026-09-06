// PrintPlan.h - the layout of a print record and who owns it.
//
// Metal has no printf in a compute kernel, so a `tt.print` appends a record
// to a device buffer that the launcher reads back and formats. The emitter
// writes the record layout as text into the module for the Python side to
// parse, so neither end hardcodes it.
//
// The buffer is device memory: threadgroup is a 32768-byte budget and a
// print buffer wants to be far larger.
#ifndef AGPU_PRINT_PLAN_H
#define AGPU_PRINT_PLAN_H

#include "agpu/plan/Elementwise.h"

#include <cstdint>
#include <string>
#include <vector>

namespace agpu {

// Not `ElemType`: a record carries 32 bits per value and narrower types
// widen on the way in, so only how to read the word is left. f16 and f32
// both arrive as `Float`.
enum class PrintType : std::int32_t {
  SInt = 0,
  UInt = 1,
  Float = 2,
};

// Floats travel as bits.
inline PrintType printTypeOf(ElemType e) {
  switch (e.kind) {
  case ElemType::Kind::Float:
    return PrintType::Float;
  case ElemType::Kind::Bool:
    return PrintType::UInt;
  case ElemType::Kind::Int:
    return e.isUnsigned ? PrintType::UInt : PrintType::SInt;
  case ElemType::Kind::Pointer:
    return PrintType::UInt;
  }
  return PrintType::SInt;
}

// The buffer is `device atomic_uint *`, so fields are counted in words. This
// order is both the emission order and the decode order.
enum class PrintField : std::int32_t {
  // Index into the host's prefix strings.
  Site = 0,
  // threadgroup_position_in_grid.x
  Pid = 1,
  // Flat thread index within the threadgroup.
  Tid = 2,
  // Coordinate along the tensor's flattened shape. Zero for a scalar.
  Index = 3,
  // A `PrintType`.
  Type = 4,
  // The value, widened to 32 bits. Floats travel as bits.
  Value = 5,
  // Which operand of the print, for sites printing several tensors that
  // share an index.
  Operand = 6,

  Count = 7,
};

inline constexpr std::int32_t kPrintRecordWords =
    static_cast<std::int32_t>(PrintField::Count);

// A field's word offset within a record.
inline constexpr std::int32_t printFieldWord(PrintField f) {
  return static_cast<std::int32_t>(f);
}

// One word: the number of records the kernel tried to write. The allocator
// bumps it unconditionally and only then checks whether the slot fits, so
// the overflow count is exact.
enum class PrintHeader : std::int32_t {
  Head = 0,

  Count = 1,
};

inline constexpr std::int32_t kPrintHeaderWords =
    static_cast<std::int32_t>(PrintHeader::Count);

inline constexpr std::int32_t printHeaderWord(PrintHeader h) {
  return static_cast<std::int32_t>(h);
}

class PrintCapacity {
public:
  PrintCapacity() = default;
  explicit PrintCapacity(std::int32_t records) : records_(records) {}

  std::int32_t records() const { return records_; }

  std::int64_t words() const {
    return (std::int64_t)kPrintHeaderWords +
           (std::int64_t)records_ * kPrintRecordWords;
  }
  std::int64_t bytes() const { return words() * 4; }

  std::int64_t wordOfRecord(std::int32_t slot) const {
    return (std::int64_t)kPrintHeaderWords +
           (std::int64_t)slot * kPrintRecordWords;
  }

private:
  std::int32_t records_ = 0;
};

// A budget: a print inside a data-dependent loop has no static record count.
// Overflow is reported.
inline constexpr std::int32_t kPrintBufferRecords = 64 * 1024;

inline PrintCapacity printCapacity() {
  return PrintCapacity(kPrintBufferRecords);
}

// One operand of one `tt.print`, as the emitter will write it.
struct PrintOperand {
  // One name per element, in print order. A scalar is one; a tensor is one
  // per register its layout gives this thread.
  std::vector<msl::Str> regs;
  ElemType elem = i32();

  // Whether the elements have a coordinate to report. A scalar has none.
  bool distributed = false;

  PrintType type() const { return printTypeOf(elem); }
};

// `site` is the index the record carries; the prefix stays on the host so
// records are fixed-width.
struct PrintSite {
  std::int32_t site = 0;
  msl::Str prefix;
  bool hex = false;
  std::vector<PrintOperand> operands;

  // Records this site writes per thread that reaches it.
  std::int32_t recordsPerThread() const {
    std::int32_t n = 0;
    for (const PrintOperand &o : operands)
      n += (std::int32_t)o.regs.size();
    // A print with no operands still says it happened.
    return n == 0 ? 1 : n;
  }
};

// The kernel's ABI depends on whether this is empty: a kernel that does not
// print takes no print buffer.
class PrintPlan {
public:
  // Sites number in add order and a body may be built twice, so a caller
  // that re-walks must clear() first.
  std::int32_t add(PrintSite site) {
    site.site = (std::int32_t)sites_.size();
    sites_.push_back(std::move(site));
    return sites_.back().site;
  }

  void clear() { sites_.clear(); }

  // Not `clear()`: an earlier kernel's sites must survive this kernel's
  // rebuild, prefixes are numbered across the whole module.
  void truncate(std::size_t n) {
    if (n < sites_.size())
      sites_.resize(n);
  }

  bool prints() const { return !sites_.empty(); }
  std::size_t siteCount() const { return sites_.size(); }
  const std::vector<PrintSite> &sites() const { return sites_; }

  PrintCapacity capacity() const { return printCapacity(); }

private:
  std::vector<PrintSite> sites_;
};

// The launcher finds the description by this string in the module, the way
// it finds the kernel name by `kernel void`.
inline constexpr const char *kPrintLayoutTag = "AGPU-PRINT-LAYOUT";

// One `key=value` per line, all decimal, so the parser needs no schema of
// its own. `sites` is the count; the prefixes follow, one per line.
inline msl::Str printLayoutText(const PrintPlan &plan) {
  const PrintCapacity cap = plan.capacity();
  msl::Str out;
  out += msl::Str(kPrintLayoutTag) +
         " headerWords=" + std::to_string(kPrintHeaderWords) + "\n";
  out += msl::Str(kPrintLayoutTag) +
         " headWord=" + std::to_string(printHeaderWord(PrintHeader::Head)) +
         "\n";
  out += msl::Str(kPrintLayoutTag) +
         " recordWords=" + std::to_string(kPrintRecordWords) + "\n";
  out += msl::Str(kPrintLayoutTag) +
         " records=" + std::to_string(cap.records()) + "\n";
  out += msl::Str(kPrintLayoutTag) + " bytes=" + std::to_string(cap.bytes()) +
         "\n";

  // The field names the host decodes by.
  const struct {
    const char *name;
    PrintField field;
  } fields[] = {
      {"site", PrintField::Site},       {"pid", PrintField::Pid},
      {"tid", PrintField::Tid},         {"index", PrintField::Index},
      {"type", PrintField::Type},       {"value", PrintField::Value},
      {"operand", PrintField::Operand},
  };
  for (const auto &f : fields)
    out += msl::Str(kPrintLayoutTag) + " field." + f.name + "=" +
           std::to_string(printFieldWord(f.field)) + "\n";

  const struct {
    const char *name;
    PrintType type;
  } types[] = {
      {"sint", PrintType::SInt},
      {"uint", PrintType::UInt},
      {"float", PrintType::Float},
  };
  for (const auto &t : types)
    out += msl::Str(kPrintLayoutTag) + " type." + t.name + "=" +
           std::to_string(static_cast<std::int32_t>(t.type)) + "\n";

  out += msl::Str(kPrintLayoutTag) +
         " sites=" + std::to_string(plan.siteCount()) + "\n";
  for (const PrintSite &s : plan.sites()) {
    // A newline would end the line the parser reads.
    msl::Str safe;
    for (const char ch : s.prefix) {
      if (ch == '\n')
        safe += "\\n";
      else if (ch == '\\')
        safe += "\\\\";
      else
        safe += ch;
    }
    // Operand count: the host labels values only when there is more than one.
    out += msl::Str(kPrintLayoutTag) + " nops." + std::to_string(s.site) + "=" +
           std::to_string(s.operands.size()) + "\n";
    out += msl::Str(kPrintLayoutTag) + " site." + std::to_string(s.site) + "." +
           (s.hex ? "hex" : "dec") + "=" + safe + "\n";
  }
  return out;
}

} // namespace agpu

#endif // AGPU_PRINT_PLAN_H
