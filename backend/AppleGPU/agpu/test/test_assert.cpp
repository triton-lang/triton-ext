// The assert record, what "halt" means and the agreement with the launcher.
#include "agpu/emit/EmitAssert.h"
#include "agpu/emit/EmitKernel.h"
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

struct Decoded {
  std::map<std::string, long long> nums;
  std::map<std::string, std::string> messages;
  std::map<std::string, std::string> wheres;

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
    const std::size_t tag = line.find("AGPU-ASSERT-LAYOUT ");
    if (tag == std::string::npos)
      continue;
    const std::string kv = line.substr(tag + 19);
    const std::size_t eq = kv.find('=');
    if (eq == std::string::npos)
      continue;
    const std::string key = kv.substr(0, eq);
    const std::string val = kv.substr(eq + 1);
    if (key.rfind("msg.", 0) == 0)
      d.messages[key.substr(4)] = val;
    else if (key.rfind("where.", 0) == 0)
      d.wheres[key.substr(6)] = val;
    else
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

AssertSite siteOf(const char *msg, const char *file, int line) {
  AssertSite s;
  s.message = msg;
  s.file = file;
  s.line = line;
  return s;
}

// The emitter folds a zero offset away, so word 0 is the bare pointer.
std::string at(const char *base, long word) {
  return word == 0 ? std::string(base)
                   : std::string(base) + " + " + std::to_string(word);
}

} // namespace

int main() {
  // ── the agreement ──────────────────────────────────────────────────────

  CASE("the MSL writer and the emitted assert description place fields "
       "identically");
  {
    AssertPlan plan;
    plan.add(siteOf("x > 0", "k.py", 12));
    const Decoded d = parseLayout(assertLayoutText(plan));
    const std::string src = helperSource(Helper::AssertRecord);

    const struct {
      const char *field;
      const char *arg;
    } fields[] = {
        {"site", "site"},
        {"pid", "pid"},
        {"tid", "tid"},
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

  CASE("the advertised byte count covers every assert record the kernel may "
       "write");
  {
    AssertPlan plan;
    plan.add(siteOf("m", "k.py", 1));
    const Decoded d = parseLayout(assertLayoutText(plan));

    const long long lastWord = d.at("headerWords") +
                               (d.at("records") - 1) * d.at("recordWords") +
                               d.at("recordWords") - 1;
    CHECK(lastWord * 4 < d.at("bytes"));
    CHECK_EQ((lastWord + 1) * 4, d.at("bytes"));
  }

  CASE("a decoded assert record round-trips through the description alone");
  {
    AssertPlan plan;
    plan.add(siteOf("cond", "k.py", 3));
    const Decoded d = parseLayout(assertLayoutText(plan));

    std::vector<unsigned> buf(
        (std::size_t)(d.at("headerWords") + 2 * d.at("recordWords")), 0u);
    auto put = [&](long long slot, const std::string &field, unsigned v) {
      buf[(std::size_t)(d.at("headerWords") + slot * d.at("recordWords") +
                        d.at("field." + field))] = v;
    };
    put(1, "site", 0);
    put(1, "pid", 9);
    put(1, "tid", 65);

    CHECK_EQ(wordAt(buf, d, 1, "pid"), 9u);
    CHECK_EQ(wordAt(buf, d, 1, "tid"), 65u);
    CHECK_EQ(wordAt(buf, d, 0, "tid"), 0u);
  }

  CASE("the message and the location reach the host");
  {
    AssertPlan plan;
    plan.add(siteOf("x > 0", "kern.py", 41));
    plan.add(siteOf("y != 0", "kern.py", 55));
    const Decoded d = parseLayout(assertLayoutText(plan));

    CHECK_EQ(d.at("sites"), 2LL);
    CHECK_EQ(d.messages.at("0"), std::string("x > 0"));
    CHECK_EQ(d.messages.at("1"), std::string("y != 0"));
    CHECK_EQ(d.wheres.at("0"), std::string("kern.py:41"));
    CHECK_EQ(d.wheres.at("1"), std::string("kern.py:55"));
  }

  CASE("a message carrying a newline does not end the line it is written on");
  {
    AssertPlan plan;
    AssertSite s = siteOf("", "k.py", 1);
    s.message = "two\nlines";
    plan.add(s);
    const std::string text = assertLayoutText(plan);
    CHECK_EQ(countOf(text, "msg.0="), 1);
    CHECK(has(text, "two\\nlines"));
  }

  CASE("an assert with no message still carries its location");
  {
    AssertPlan plan;
    plan.add(siteOf("", "kern.py", 88));
    const Decoded d = parseLayout(assertLayoutText(plan));
    CHECK_EQ(d.messages.at("0"), std::string(""));
    CHECK_EQ(d.wheres.at("0"), std::string("kern.py:88"));
  }

  // ── what "halt" means ──────────────────────────────────────────────────

  CASE("a failing thread records before it returns");
  {
    msl::Context c;
    KernelNames nm;
    msl::Block b;
    emitAssert(c, b, siteOf("m", "k.py", 1), {"r0"}, nm);
    const std::string out = render(b);

    const std::size_t rec = out.find("__agpu_assert_record");
    const std::size_t ret = out.find("return");
    CHECK(rec != std::string::npos);
    CHECK(ret != std::string::npos);
    CHECK(rec < ret);
  }

  CASE("an assert with a barrier after it records but does not return");
  {
    AssertContext after;
    after.barrierFollows = true;
    CHECK(assertHaltFor(after) == AssertHalt::Continue);

    AssertContext clear;
    CHECK(assertHaltFor(clear) == AssertHalt::Return);

    msl::Context c;
    KernelNames nm;
    AssertSite s = siteOf("m", "k.py", 1);
    s.halt = AssertHalt::Continue;
    msl::Block b;
    emitAssert(c, b, s, {"r0"}, nm);
    const std::string out = render(b);

    CHECK(has(out, "__agpu_assert_record"));
    CHECK(!has(out, "return"));
  }

  CASE("the assert fires when any register is false");
  {
    msl::Context c;
    KernelNames nm;
    msl::Block b;
    emitAssert(c, b, siteOf("m", "k.py", 1), {"r0", "r1", "r2"}, nm);
    const std::string out = render(b);

    CHECK(has(out, "!r0 || !r1 || !r2"));
    CHECK_EQ(countOf(out, "__agpu_assert_record"), 1);
  }

  CASE("a record names the thread that failed");
  {
    msl::Context c;
    KernelNames nm;
    msl::Block b;
    emitAssert(c, b, siteOf("m", "k.py", 1), {"r0"}, nm);
    const std::string out = render(b);
    CHECK(has(out, nm.threadgroupPos + ".x"));
    CHECK(has(out, nm.threadId + ".x"));
    CHECK(has(out, nm.assertBuffer));
  }

  CASE("the head counts failures, so a truncated run still says how many");
  {
    const std::string src = helperSource(Helper::AssertRecord);
    const std::size_t bump = src.find("atomic_fetch_add_explicit");
    const std::size_t test = src.find("slot >=");
    CHECK(bump != std::string::npos);
    CHECK(test != std::string::npos);
    CHECK(bump < test);
  }

  CASE("the assert buffer is far smaller than the print buffer");
  {
    CHECK(assertCapacity().records() < printCapacity().records());
    CHECK(assertCapacity().bytes() < printCapacity().bytes());
  }

  // ── the ABI ────────────────────────────────────────────────────────────

  CASE("an assert buffer never lands on the print buffer's binding");
  {
    const std::vector<KernelArg> args{KernelArg{"a", f32(), true}};

    const KernelAbi printsOnly = planKernelAbi(
        args, 4, DebugChannels{DebugBinding::Bound, DebugBinding::None});
    const KernelAbi assertsOnly = planKernelAbi(
        args, 4, DebugChannels{DebugBinding::None, DebugBinding::Bound});
    const KernelAbi both = planKernelAbi(
        args, 4, DebugChannels{DebugBinding::Bound, DebugBinding::Bound});

    CHECK(printsOnly.hasPrintBuffer);
    CHECK(!printsOnly.hasAssertBuffer);
    CHECK(assertsOnly.hasAssertBuffer);
    CHECK(!assertsOnly.hasPrintBuffer);

    CHECK(both.printBufferIndex != both.assertBufferIndex);
    CHECK(both.printBufferIndex < both.assertBufferIndex);
    CHECK_EQ(both.bufferCount, printsOnly.bufferCount + 1);

    for (const KernelAbi *abi : {&printsOnly, &assertsOnly, &both})
      CHECK_EQ(abi->placements[0].index, 0LL);
  }

  CASE("the signature declares the buffer a failing thread writes through");
  {
    msl::Context c;
    KernelFacts f;
    f.name = "k";
    f.args = {KernelArg{"a", f32(), true}};
    f.numWarps = 4;
    f.debug.assertion = DebugBinding::Bound;

    KernelNames nm;
    const KernelResult r = emitKernel(c, f, [&](msl::Context &cc, bool) {
      msl::Block b;
      emitAssert(cc, b, siteOf("m", "k.py", 1), {"a"}, nm);
      return b;
    });
    CHECK(r.ok());

    bool declared = false;
    for (const msl::Function::Param &p : r.fn->params)
      if (p.name == nm.assertBuffer)
        declared = true;
    CHECK(declared);
  }

  CASE("a kernel with no assert declares no assert parameter");
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
      CHECK(p.name != KernelNames{}.assertBuffer);
  }

  CASE("a body built twice numbers its assert sites the same both times");
  {
    AssertPlan plan;
    plan.add(siteOf("earlier kernel", "k.py", 1));
    const std::size_t mark = plan.siteCount();

    for (int build = 0; build < 2; ++build) {
      plan.truncate(mark);
      CHECK_EQ(plan.add(siteOf("first", "k.py", 2)), 1);
      CHECK_EQ(plan.add(siteOf("second", "k.py", 3)), 2);
    }
    CHECK_EQ(plan.siteCount(), (std::size_t)3);

    const Decoded d = parseLayout(assertLayoutText(plan));
    CHECK_EQ(d.messages.at("0"), std::string("earlier kernel"));
    CHECK_EQ(d.messages.at("2"), std::string("second"));
  }

  CASE("the record helper is defined wherever it is named");
  {
    HelperSet h;
    h.add(Helper::AssertRecord);
    CHECK(h.has(Helper::AssertRecord));
    CHECK(!helperSource(Helper::AssertRecord).empty());

    std::ostringstream os;
    printPrelude(os, h, /*header=*/false);
    CHECK(has(os.str(), helperName(Helper::AssertRecord)));
  }

  CASE("the assert buffer is device memory, so it never touches the pool "
       "budget");
  {
    CHECK(has(helperSource(Helper::AssertRecord), "device atomic_uint *"));
    CHECK(!has(helperSource(Helper::AssertRecord), "threadgroup"));
  }

  return ::agpu_test::report("Assert");
}
