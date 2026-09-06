// The print record: the offsets the MSL writer emits must match the layout
// description.
#include "agpu/emit/EmitKernel.h"
#include "agpu/emit/EmitPrint.h"
#include "agpu/emit/PrintModule.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <map>
#include <sstream>

using namespace agpu;
using agpu_test::countOf;
using agpu_test::has;
using agpu_test::render;

namespace {

// Decodes using only the parsed description, naming no constant from
// PrintPlan.h.
struct Decoded {
  std::map<std::string, long long> nums;
  std::map<std::string, std::string> prefixes;
  std::map<std::string, bool> hex;

  long long at(const std::string &k) const {
    auto it = nums.find(k);
    return it == nums.end() ? -1 : it->second;
  }
};

Decoded parseLayout(const std::string &text) {
  Decoded d;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t tag = line.find("AGPU-PRINT-LAYOUT ");
    if (tag == std::string::npos)
      continue;
    const std::string kv = line.substr(tag + 18);
    const std::size_t eq = kv.find('=');
    if (eq == std::string::npos)
      continue;
    const std::string key = kv.substr(0, eq);
    const std::string val = kv.substr(eq + 1);
    if (key.rfind("site.", 0) == 0) {
      const std::size_t dot = key.find('.', 5);
      const std::string idx = key.substr(5, dot - 5);
      d.prefixes[idx] = val;
      d.hex[idx] = key.substr(dot + 1) == "hex";
      continue;
    }
    d.nums[key] = std::stoll(val);
  }
  return d;
}

unsigned wordAt(const std::vector<unsigned> &buf, const Decoded &d,
                long long slot, const std::string &field) {
  const long long base =
      d.at("headerWords") + slot * d.at("recordWords") + d.at("field." + field);
  return buf[(std::size_t)base];
}

PrintSite siteOf(std::int32_t n, const char *prefix,
                 std::vector<PrintOperand> ops) {
  PrintSite s;
  s.site = n;
  s.prefix = prefix;
  s.operands = std::move(ops);
  return s;
}

PrintOperand operandOf(std::vector<msl::Str> regs, ElemType e) {
  PrintOperand o;
  o.regs = std::move(regs);
  o.elem = e;
  return o;
}

PrintOperand tensorOf(std::vector<msl::Str> regs, ElemType e) {
  PrintOperand o = operandOf(std::move(regs), e);
  o.distributed = true;
  return o;
}

PrintIndexFn literalIndex(msl::Context &c) {
  return [&c](std::size_t, std::size_t r) { return c.lit((std::int64_t)r); };
}

// The emitter folds a zero offset away, so word 0 is the bare pointer.
std::string at(const char *base, long word) {
  return word == 0 ? std::string(base)
                   : std::string(base) + " + " + std::to_string(word);
}

} // namespace

int main() {
  CASE("the MSL writer and the emitted print description place fields "
       "identically");
  {
    PrintPlan plan;
    plan.add(siteOf(0, "x", {operandOf({"r0"}, i32())}));
    const Decoded d = parseLayout(printLayoutText(plan));
    const std::string src = helperSource(Helper::PrintAppend);

    // Each store lands at `rec + <word>` for its field.
    const struct {
      const char *field;
      const char *arg;
    } fields[] = {
        {"site", "site"},       {"pid", "pid"},   {"tid", "tid"},
        {"index", "index"},     {"type", "type"}, {"value", "value"},
        {"operand", "operand"},
    };
    for (const auto &f : fields) {
      const std::string store =
          "atomic_store_explicit(" +
          at("rec", d.at(std::string("field.") + f.field)) + ", " + f.arg + ",";
      CHECK(has(src, store));
    }

    CHECK(has(src, "buf + " + std::to_string(d.at("headerWords")) +
                       " + slot * " + std::to_string(d.at("recordWords"))));
    CHECK(has(src, at("buf", d.at("headWord")) + ", 1u"));
    CHECK(has(src, "slot >= " + std::to_string(d.at("records")) + "u"));
  }

  CASE("the advertised byte count covers every print record the kernel may "
       "write");
  {
    PrintPlan plan;
    plan.add(siteOf(0, "x", {}));
    const Decoded d = parseLayout(printLayoutText(plan));

    const long long lastWord = d.at("headerWords") +
                               (d.at("records") - 1) * d.at("recordWords") +
                               d.at("recordWords") - 1;
    CHECK(lastWord * 4 < d.at("bytes"));
    CHECK_EQ((lastWord + 1) * 4, d.at("bytes"));
  }

  CASE("a decoded print record round-trips through the description alone");
  {
    PrintPlan plan;
    plan.add(siteOf(0, "v", {operandOf({"r0"}, f32())}));
    const Decoded d = parseLayout(printLayoutText(plan));

    std::vector<unsigned> buf(
        (std::size_t)(d.at("headerWords") + 2 * d.at("recordWords")), 0u);
    auto put = [&](long long slot, const std::string &field, unsigned v) {
      buf[(std::size_t)(d.at("headerWords") + slot * d.at("recordWords") +
                        d.at("field." + field))] = v;
    };
    put(1, "site", 0);
    put(1, "pid", 7);
    put(1, "tid", 33);
    put(1, "index", 2);
    put(1, "type", (unsigned)d.at("type.float"));
    put(1, "value", 0x40490fdbu); // 3.14159274f

    CHECK_EQ(wordAt(buf, d, 1, "pid"), 7u);
    CHECK_EQ(wordAt(buf, d, 1, "tid"), 33u);
    CHECK_EQ(wordAt(buf, d, 1, "index"), 2u);
    CHECK_EQ(wordAt(buf, d, 1, "type"), (unsigned)d.at("type.float"));
    CHECK_EQ(wordAt(buf, d, 0, "value"), 0u);
  }

  CASE("a float travels as bits and an integer as a value");
  {
    msl::Context c;
    KernelNames nm;
    msl::Block b;
    emitPrint(c, b, siteOf(0, "f", {operandOf({"r0"}, f32())}), literalIndex(c),
              nm);
    CHECK(has(render(b), "as_type<uint>(r0)"));

    msl::Block ints;
    emitPrint(c, ints, siteOf(0, "i", {operandOf({"r0"}, i32())}),
              literalIndex(c), nm);
    CHECK(!has(render(ints), "as_type"));
    CHECK(has(render(ints), "(uint)r0"));
  }

  CASE("a narrow float widens before it is reinterpreted");
  {
    // `as_type<uint>` on a half is a 16-bit reinterpret.
    msl::Context c;
    KernelNames nm;
    msl::Block b;
    emitPrint(c, b, siteOf(0, "h", {operandOf({"r0"}, f16())}), literalIndex(c),
              nm);
    const std::string out = render(b);
    CHECK(has(out, "as_type<uint>((float)r0)"));
  }

  CASE("a signed and an unsigned integer are told apart");
  {
    CHECK(printTypeOf(i32()) == PrintType::SInt);
    ElemType u32 = i32();
    u32.isUnsigned = true;
    CHECK(printTypeOf(u32) == PrintType::UInt);
    CHECK(printTypeOf(f16()) == PrintType::Float);
    CHECK(printTypeOf(bf16()) == PrintType::Float);
  }

  CASE("one append per element, each carrying the index it was given");
  {
    msl::Context c;
    KernelNames nm;
    msl::Block b;
    emitPrint(
        c, b,
        siteOf(0, "pair",
               {tensorOf({"a0", "a1"}, i32()), tensorOf({"b0"}, i32())}),
        [&c](std::size_t o, std::size_t r) {
          return c.var("ix" + std::to_string(o) + std::to_string(r));
        },
        nm);
    const std::string out = render(b);
    CHECK_EQ(countOf(out, "__agpu_print_append"), 3);
    CHECK(has(out, "ix00"));
    CHECK(has(out, "ix01"));
    CHECK(has(out, "ix10"));
  }

  CASE("a scalar reports no element index");
  {
    msl::Context c;
    KernelNames nm;
    PrintOperand scalar = operandOf({"s"}, i32());
    CHECK(!scalar.distributed);
    CHECK(tensorOf({"t"}, i32()).distributed);
  }

  CASE("a print with no operands still writes a record");
  {
    msl::Context c;
    KernelNames nm;
    msl::Block b;
    emitPrint(c, b, siteOf(3, "here", {}), literalIndex(c), nm);
    CHECK_EQ(countOf(render(b), "__agpu_print_append"), 1);
    CHECK_EQ(PrintSite{}.recordsPerThread(), 1);
  }

  CASE("a record names the thread that wrote it");
  {
    msl::Context c;
    KernelNames nm;
    msl::Block b;
    emitPrint(c, b, siteOf(0, "x", {operandOf({"r0"}, i32())}), literalIndex(c),
              nm);
    const std::string out = render(b);
    CHECK(has(out, nm.threadgroupPos + ".x"));
    CHECK(has(out, nm.threadId + ".x"));
    CHECK(has(out, nm.printBuffer));
  }

  CASE("each site is numbered, so the host knows which prefix to print");
  {
    PrintPlan plan;
    CHECK(!plan.prints());
    CHECK_EQ(plan.add(siteOf(0, "first", {})), 0);
    CHECK_EQ(plan.add(siteOf(0, "second", {})), 1);
    CHECK(plan.prints());

    const Decoded d = parseLayout(printLayoutText(plan));
    CHECK_EQ(d.at("sites"), 2LL);
    CHECK_EQ(d.prefixes.at("0"), std::string("first"));
    CHECK_EQ(d.prefixes.at("1"), std::string("second"));
  }

  CASE("a prefix carrying a newline does not end the line it is written on");
  {
    // The description is line-oriented.
    PrintPlan plan;
    PrintSite s = siteOf(0, "", {});
    s.prefix = "two\nlines";
    plan.add(s);
    const std::string text = printLayoutText(plan);
    CHECK_EQ(countOf(text, "site.0."), 1);
    CHECK(has(text, "two\\nlines"));

    const Decoded d = parseLayout(text);
    CHECK_EQ(d.prefixes.at("0"), std::string("two\\nlines"));
  }

  CASE("the hex flag reaches the host, because it changes what is printed");
  {
    PrintPlan plan;
    PrintSite s = siteOf(0, "h", {});
    s.hex = true;
    plan.add(s);
    const Decoded d = parseLayout(printLayoutText(plan));
    CHECK(d.hex.at("0"));

    PrintPlan dec;
    dec.add(siteOf(0, "d", {}));
    CHECK(!parseLayout(printLayoutText(dec)).hex.at("0"));
  }

  CASE("a kernel that does not print takes no print buffer");
  {
    const std::vector<KernelArg> args{KernelArg{"a", f32(), true}};
    const KernelAbi none = planKernelAbi(args, 4, DebugChannels{});
    CHECK(!none.hasPrintBuffer);
    CHECK_EQ(none.bufferCount, 1LL);
  }

  CASE("the print buffer binds after everything the host already places");
  {
    // The launcher's pointer arguments are positional.
    const std::vector<KernelArg> args{KernelArg{"a", f32(), true},
                                      KernelArg{"b", f32(), true},
                                      KernelArg{"n", i32(), false}};
    const KernelAbi without = planKernelAbi(args, 4, DebugChannels{});
    const KernelAbi with = planKernelAbi(
        args, 4, DebugChannels{DebugBinding::Bound, DebugBinding::None});

    for (std::size_t i = 0; i < args.size(); ++i)
      CHECK_EQ(with.placements[i].index, without.placements[i].index);
    CHECK_EQ(with.argBufferIndex, without.argBufferIndex);

    CHECK(with.hasPrintBuffer);
    CHECK_EQ(with.printBufferIndex, without.bufferCount);
    CHECK_EQ(with.bufferCount, without.bufferCount + 1);
  }

  CASE("the argument buffer keeps its index when a print buffer follows it");
  {
    const std::vector<KernelArg> args{KernelArg{"a", f32(), true},
                                      KernelArg{"n", i32(), false}};
    const KernelAbi abi = planKernelAbi(
        args, 4, DebugChannels{DebugBinding::Bound, DebugBinding::None});
    CHECK_EQ(abi.argBufferIndex, 1LL);
    CHECK_EQ(abi.printBufferIndex, 2LL);
    CHECK(abi.argBufferIndex != abi.bufferCount - 1);
  }

  CASE("the signature declares the buffer the appends write through");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {KernelArg{"a", f32(), true}};
    f.numWarps = 4;
    f.debug.print = DebugBinding::Bound;

    KernelNames nm;
    const KernelResult r = emitKernel(c, f, [&](msl::Context &cc, bool) {
      msl::Block b;
      emitPrint(cc, b, siteOf(0, "x", {operandOf({"a"}, f32())}),
                literalIndex(cc), nm);
      return b;
    });
    CHECK(r.ok());

    bool declared = false;
    for (const msl::Function::Param &p : r.fn->params)
      if (p.name == nm.printBuffer)
        declared = true;
    CHECK(declared);
  }

  CASE("a kernel with no print declares no print parameter");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {KernelArg{"a", f32(), true}};
    f.numWarps = 4;

    const KernelResult r =
        emitKernel(c, f, [](msl::Context &, bool) { return msl::Block{}; });
    CHECK(r.ok());
    for (const msl::Function::Param &p : r.fn->params)
      CHECK(p.name != KernelNames{}.printBuffer);
  }

  CASE("a body built twice numbers its print sites the same both times");
  {
    // `emitKernel` builds a body twice: once to measure, again for the rolled
    // form.
    PrintPlan plan;
    plan.add(siteOf(0, "earlier kernel", {}));
    const std::size_t mark = plan.siteCount();

    for (int build = 0; build < 2; ++build) {
      plan.truncate(mark);
      CHECK_EQ(plan.add(siteOf(0, "first", {})), 1);
      CHECK_EQ(plan.add(siteOf(0, "second", {})), 2);
    }
    CHECK_EQ(plan.siteCount(), (std::size_t)3);

    // The module numbers prefixes across all its kernels.
    const Decoded d = parseLayout(printLayoutText(plan));
    CHECK_EQ(d.prefixes.at("0"), std::string("earlier kernel"));
    CHECK_EQ(d.prefixes.at("1"), std::string("first"));
    CHECK_EQ(d.prefixes.at("2"), std::string("second"));
  }

  CASE("a description is emitted exactly when a buffer is bound");
  {
    PrintPlan silent;
    CHECK(!silent.prints());
    CHECK_EQ(silent.siteCount(), (std::size_t)0);

    PrintPlan loud;
    loud.add(siteOf(0, "x", {}));
    CHECK(loud.prints());

    const Decoded d = parseLayout(printLayoutText(loud));
    for (const char *key :
         {"headerWords", "headWord", "recordWords", "records", "bytes",
          "field.site", "field.pid", "field.tid", "field.index", "field.type",
          "field.value", "field.operand", "type.sint", "type.uint",
          "type.float", "sites"})
      CHECK(d.at(key) >= 0);
  }

  CASE("the append helper is defined wherever it is named");
  {
    HelperSet h;
    h.add(Helper::PrintAppend);
    CHECK(h.has(Helper::PrintAppend));
    CHECK(!helperSource(Helper::PrintAppend).empty());

    std::ostringstream os;
    printPrelude(os, h, /*header=*/false);
    CHECK(has(os.str(), helperName(Helper::PrintAppend)));
  }

  CASE(
      "the print buffer is device memory, so it never touches the pool budget");
  {
    CHECK(has(helperSource(Helper::PrintAppend), "device atomic_uint *"));
    CHECK(!has(helperSource(Helper::PrintAppend), "threadgroup"));
  }

  CASE("the head counts attempts, so what was lost can be reported");
  {
    // The bump precedes the bounds test.
    const std::string src = helperSource(Helper::PrintAppend);
    const std::size_t bump = src.find("atomic_fetch_add_explicit");
    const std::size_t test = src.find("slot >=");
    CHECK(bump != std::string::npos);
    CHECK(test != std::string::npos);
    CHECK(bump < test);
  }

  return ::agpu_test::report("Print");
}
