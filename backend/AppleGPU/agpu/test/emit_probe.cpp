// Not a unit test: a generator whose output is fed to the Metal compiler by
// metal_compiles.sh.
#include "agpu/Emitter.h"
#include "agpu/bind/SymbolTable.h"
#include "agpu/emit/EmitAtomic.h"
#include "agpu/emit/EmitCas.h"
#include "agpu/emit/EmitControl.h"
#include "agpu/emit/EmitDeviceFn.h"
#include "agpu/emit/EmitDot.h"
#include "agpu/emit/EmitElementwise.h"
#include "agpu/emit/EmitEpilogue.h"
#include "agpu/emit/EmitKernel.h"
#include "agpu/emit/EmitMemDesc.h"
#include "agpu/emit/EmitMove.h"
#include "agpu/emit/EmitPanel.h"
#include "agpu/emit/EmitPoll.h"
#include "agpu/emit/EmitRange.h"
#include "agpu/emit/EmitReduce.h"
#include "agpu/emit/EmitRegion.h"
#include "agpu/emit/EmitScan.h"
#include "agpu/emit/Prelude.h"
#include "agpu/msl/Printer.h"

#include <iostream>
#include <limits>
#include <sstream>
#include <string>

using namespace agpu;

namespace {

void printHelperUses(std::ostream &os) {
  os << R"(kernel void agpu_probe_helpers(device float *out [[buffer(0)]],
                               device atomic_uint *word [[buffer(1)]],
                               uint tid [[thread_position_in_grid]]) {
  float v = out[tid];
  out[tid] = __agpu_erf(v)
           + float(__agpu_rtz_half(v))
           + float(__agpu_rtne_half(v))
           + float(__agpu_rtz_bfloat(v))
           + float(__agpu_rtne_bfloat(v))
           + float(__agpu_rtne_int_half(v))
           + float(__agpu_rtne_int_bfloat(v))
           + __agpu_e4m3_to_f32(__agpu_f32_to_e4m3(v))
           + __agpu_e5m2_to_f32(__agpu_f32_to_e5m2(v))
           + __agpu_atomic_rmw_f32(word, v, 0)
           + float(__agpu_atomic_rmw_packed16<half>(word, true, v, 0))
           + float(__agpu_atomic_rmw_packed16<bfloat>(word, false, v, 1));
}
)";
}

void printEmittedKernel(std::ostream &os, HelperSet &helpers) {
  msl::Context c;
  KernelFacts f;
  f.name = "agpu_probe_kernel";
  f.args = {{"out", f32(), true},
            {"a", f32(), true},
            {"b", f32(), true},
            {"n", i32(), false},
            {"alpha", f32(), false}};
  f.numWarps = 4;
  f.poolBytes = 4096;

  AtomicFacts af;
  af.op = RmwOp::Max;
  af.elem = ElemClass::Float;
  af.bits = 32;
  const AtomicPlan ap = planAtomic(af, MemOrder::Relaxed);
  helpers.require(ap);

  KernelResult r = emitKernel(c, f, [&](msl::Context &cc, bool) {
    msl::Block b;

    MoveSite site;
    site.elem = [&cc](int64_t r) {
      return cc.subscript(cc.var("a"), cc.lit(r));
    };
    for (int i = 0; i < 4; ++i)
      site.values.push_back("va" + std::to_string(i));

    MoveFacts mf;
    mf.regCount = 4;
    mf.elemBits = 32;
    mf.ptr = PtrDims(2, PtrInfo{4, 4});
    mf.coherent = true;
    for (int i = 0; i < 2; ++i)
      mf.bases.push_back({0, 1 << i});
    emitMove(cc, b, mf, planMove(mf), site, f32());

    AtomicNames an;
    b.push_back(cc.declStmt(msl::Type::scalar(msl::Scalar::F32), an.result,
                            cc.litF(0.0)));
    const msl::Type atomicPtr =
        msl::Type::named("atomic_uint").pointerTo(msl::AddrSpace::Device);
    b.push_back(
        cc.declStmt(atomicPtr, "aw", cc.cast(atomicPtr, cc.var("out"))));
    emitAtomic(cc, b, ap, "aw", "va0", an);
    return b;
  });

  msl::Block blk{r.fn};
  msl::Printer p(os);
  p.printBlock(blk);
}

// A panel tile whose M ends mid-fragment: 60 rows is seven whole fragments
// plus an edge computed at full 8x8 width, with surplus rows withheld by the
// readback guard.
void printRaggedPanel(std::ostream &os) {
  msl::Context c;
  PanelNames nm;
  DotFacts f;
  f.M = 60;
  f.N = 32;
  f.K = 16;
  f.aElemBytes = 2;
  f.bElemBytes = 2;
  f.numWarps = 1;

  const Panel pan = panelCost(64, 32, 16, 2, kAccBytes);
  const PanelSchedule s = planPanelSchedule(f, pan);
  const PanelTile &t = s.tiles[0];

  CoordSource cs;
  LayoutBasis row, col;
  row.lane = {1, 2, 4, 8, 16};
  col.lane = {0, 0, 0, 0, 0};
  cs.dims = {row, col};

  msl::SmallVec<StageAction, 8> aAct, cAct;
  msl::SmallVec<msl::Str, 8> aNm, cNm, cBase;
  for (int r = 0; r < 4; ++r) {
    aNm.push_back("a" + std::to_string(r));
    if (auto a =
            planStage(r, {{0, 8 * r, 8 * r + 7}, {1, 0, 15}},
                      {{0, 0, t.m.size()}, {1, 0, t.k.size()}}, {8 * r, 0}))
      aAct.push_back(*a);
  }
  for (int r = 0; r < 8; ++r) {
    cNm.push_back("cr" + std::to_string(r));
    cBase.push_back("");
    if (auto a = planStage(r, {{0, 8 * r, 8 * r + 7}, {1, 0, 31}},
                           t.readbackWindows(), {8 * r, 0}))
      cAct.push_back(*a);
  }

  PanelInputs in;
  in.a = OperandSource{PanelNames{}.poolA, Stride(t.aView().strideAt(0))};
  in.aActions = aAct;
  in.aNames = aNm;
  in.cActions = cAct;
  in.cNames = cNm;
  in.cBases = cBase;

  msl::Block body;
  const WarpGrid g = panelWarpGrid(t, f.numWarps);
  emitPanelTile(c, body, t, nm, in, PanelCoords::forAll(cs), g,
                planWarpProgram(g));

  os << "kernel void agpu_probe_ragged(device float *out [[buffer(0)]],\n"
        "                              uint3 tid [[thread_position_in_"
        "threadgroup]]) {\n"
        "  threadgroup half pA["
     << t.aView().cosizeElems()
     << "];\n"
        "  threadgroup half pB["
     << t.bView().cosizeElems()
     << "];\n"
        "  threadgroup float pC["
     << t.cView().cosizeElems()
     << "];\n"
        "  int lane = tid.x & 31;\n";
  for (const msl::Str &n : aNm)
    os << "  half " << n << " = 0;\n";
  for (const msl::Str &n : cNm)
    os << "  float " << n << " = 0;\n";

  msl::Printer p(os);
  p.printBlock(body);

  os << "  out[tid.x] = ";
  for (std::size_t i = 0; i < cNm.size(); ++i)
    os << (i ? " + " : "") << cNm[i];
  os << ";\n}\n";
}

void printElementwise(std::ostream &os) {
  msl::Context c;
  msl::Block body;

  const FCmp all[] = {FCmp::False, FCmp::OEq, FCmp::OGt, FCmp::OGe,
                      FCmp::OLt,   FCmp::OLe, FCmp::ONe, FCmp::Ord,
                      FCmp::UEq,   FCmp::UGt, FCmp::UGe, FCmp::ULt,
                      FCmp::ULe,   FCmp::UNe, FCmp::Uno, FCmp::True};
  int i = 0;
  for (FCmp pred : all)
    body.push_back(emitFCmp(c, pred, "p" + std::to_string(i++), "fa", "fb"));

  body.push_back(
      c.declStmt(msl::Type::scalar(msl::Scalar::F32), "mn",
                 minMaxExpr(c, MathFn2::Min, f32(), "fa", "fb", true)));
  body.push_back(
      c.declStmt(msl::Type::scalar(msl::Scalar::F32), "mx",
                 minMaxExpr(c, MathFn2::Max, f32(), "fa", "fb", false)));
  body.push_back(emitMath(c, MathFn::Isnan, f32(), "nn", c.var("fa")));
  body.push_back(emitMath(c, MathFn::Exp10, f32(), "e10", c.var("fa")));

  {
    int i = 0;
    for (const MathSpelling &s : kMathSpellings)
      body.push_back(
          emitMath(c, s.fn, f32(), "mfn" + std::to_string(i++), c.var("fa")));
  }
  body.push_back(
      c.declStmt(msl::Type::scalar(msl::Scalar::F32), "fm",
                 mathExpr(c, MathFn2::Fmod, c.var("fa"), c.var("fb"))));
  body.push_back(c.declStmt(
      msl::Type::scalar(msl::Scalar::F32), "fu",
      mathExpr(c, MathFn3::Fma, c.var("fa"), c.var("fb"), c.var("fa"))));
  body.push_back(c.declStmt(
      msl::Type::scalar(msl::Scalar::F32), "cl",
      mathExpr(c, MathFn3::Clamp, c.var("fa"), c.litF(0.0), c.litF(1.0))));
  body.push_back(
      c.declStmt(msl::Type::scalar(msl::Scalar::I32), "mh",
                 mathExpr(c, MathFn2::Mulhi, c.var("ia"), c.var("ib"))));

  body.push_back(emitEw(c, EwOp::DivU, i32(), "du", c.var("ia"), c.var("ib")));
  body.push_back(emitEw(c, EwOp::ShrU, i32(), "su", c.var("ia"), c.var("ib")));
  body.push_back(
      emitEw(c, EwOp::CmpGeU, i32(), "ge", c.var("ia"), c.var("ib")));
  body.push_back(emitEw(c, EwOp::DivF, f32(), "df", c.var("fa"), c.var("fb")));

  body.push_back(
      c.declStmt(msl::Type::scalar(msl::Scalar::U32), "bits",
                 c.bitcast(msl::Type::scalar(msl::Scalar::U32), c.var("fa"))));
  body.push_back(c.declStmt(
      msl::Type::scalar(msl::Scalar::F16), "hv",
      c.construct(msl::Type::scalar(msl::Scalar::F16), c.var("fa"))));
  body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::I32), "sv",
                            c.cast(msl::Type::scalar(msl::Scalar::I32),
                                   c.var("fa"), msl::Cast::Style::Static)));

  body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::F32), "big",
                            c.litF(std::numeric_limits<double>::infinity())));
  body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::F32), "neg",
                            c.litF(-std::numeric_limits<double>::infinity())));
  body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::F32), "non",
                            c.litF(std::numeric_limits<double>::quiet_NaN())));

  body.push_back(c.arrayDecl(
      msl::Type::scalar(msl::Scalar::I32), "tbl",
      msl::SmallVec<msl::Expr *, 4>{c.lit(1), c.lit(2), c.lit(3), c.lit(4)}));

  body.push_back(
      emitSelect(c, f32(), "sel", c.var("p1"), c.var("fa"), c.var("fb")));

  os << "kernel void agpu_probe_elementwise(device float *out [[buffer(0)]],\n"
        "                                   device int *iout [[buffer(1)]],\n"
        "                                   uint t [[thread_position_in_grid]])"
        " {\n"
        "  float fa = out[t];\n"
        "  float fb = out[t + 1];\n"
        "  int ia = iout[t];\n"
        "  int ib = iout[t + 1];\n";

  msl::Printer p(os);
  p.printBlock(body);

  os << "  out[t] = mn + mx + e10 + fm + fu + cl + df + sel + hv + big + neg"
        " + non\n"
        "         + (float)nn + (float)bits + (float)sv;\n"
        "  iout[t] = mh + du + su + tbl[t & 3] + (int)ge";
  for (int j = 0; j < (int)(sizeof(all) / sizeof(all[0])); ++j)
    os << " + (int)p" << j;
  os << ";\n}\n";
}

void printPlannedDot(std::ostream &os) {
  msl::Context c;

  DotFacts f;
  f.M = 64;
  f.N = 64;
  f.K = 64;
  f.aElemBytes = 2;
  f.bElemBytes = 2;
  f.numWarps = 4;

  const Plan p = planDot(f, Bytes(kTGResidentBudgetBytes));

  DotInputs in;
  in.a = {"pA", 64};
  in.b = {"pB", 64};

  msl::Block body;
  const Decision d = emitDot(c, body, p, in);
  if (!d.ok())
    return;

  os << "kernel void agpu_probe_dot(device float *out [[buffer(0)]],\n"
        "                           uint3 tid "
        "[[thread_position_in_threadgroup]]) {\n"
        "  threadgroup half pA[4096];\n"
        "  threadgroup half pB[4096];\n"
        "  threadgroup float pC[4096];\n"
        "  int warp = tid.x / 32;\n";
  msl::Printer pr(os);
  pr.printBlock(body);
  os << "  out[tid.x] = pC[tid.x];\n}\n";
}

// AGX computes an i64 induction variable in the Gauss-sum closed form at i65
// intermediate width and gets it wrong, hence the trip-count form here.
void printLoops(std::ostream &os) {
  msl::Context c;

  msl::Block wide;
  LoopBounds wb;
  wb.iv = "i64v";
  wb.lo = c.lit(0);
  wb.hi = c.var("n");
  wb.step = c.lit(1);
  wb.wideIv = true;
  emitFor(c, wide, wb, {}, {},
          msl::Block{c.assignOp(msl::BinOp::Add, c.var("sum"), c.var("i64v"))},
          {});

  msl::Block runtime;
  LoopBounds rb;
  rb.iv = "k";
  rb.lo = c.lit(0);
  rb.hi = c.var("n");
  rb.step = c.var("bk");
  emitFor(c, runtime, rb, {}, {},
          msl::Block{c.assignOp(msl::BinOp::Add, c.var("sum"), c.var("k"))},
          {});

  os << "kernel void agpu_probe_loops(device long *out [[buffer(0)]],\n"
        "                             constant int &n [[buffer(1)]],\n"
        "                             constant int &bk [[buffer(2)]],\n"
        "                             uint t [[thread_position_in_grid]]) {\n"
        "  long sum = 0;\n";
  msl::Printer p(os);
  p.printBlock(wide);
  p.printBlock(runtime);
  os << "  out[t] = sum;\n}\n";
}

void printIntegerReduce(std::ostream &os) {
  msl::Context c;
  ReduceNames rnm;
  rnm.scratch = {"pscr0"};
  ScanNames snm;

  auto adder = [&c](msl::Block &b, const msl::SmallVec<msl::Str, 4> &a,
                    const msl::SmallVec<msl::Str, 4> &p) {
    msl::SmallVec<msl::Str, 4> out;
    for (std::size_t k = 0; k < a.size(); ++k) {
      const msl::Str n =
          "t" + std::to_string(k) + "_" + std::to_string((int)b.size());
      b.push_back(
          c.declStmt(msl::Type::scalar(msl::Scalar::I32), n,
                     c.binary(msl::BinOp::Add, c.var(a[k]), c.var(p[k]))));
      out.push_back(n);
    }
    return out;
  };

  ReductionPlan rp;
  ReductionGroup g;
  g.key = CoordKey({0});
  g.sourceRegs = {0, 1};
  rp.groups.push_back(g);
  rp.laneSteps = laneStepsFromMask(0b11111);
  rp.warpSubset = subsetsOf(0b11, 4);
  rp.warpMask = 0b11;
  rp.scratch = ScratchLayout{threadsFor(4), kWarpSize};
  rp.elems = {i32()};

  msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> rsrc;
  rsrc.push_back({"r0", "r1"});

  msl::Block rbody;
  emitReduce(c, rbody, rp, 4, rsrc, rnm, adder);

  ScanFacts sf;
  sf.laneBits = {{0, 1}, {1, 2}};
  sf.numWarps = 1;
  sf.regCount = 1;
  sf.elems = {i32()};
  const ScanPlan sp = planScan(sf);

  msl::SmallVec<msl::SmallVec<msl::Str, 8>, 4> ssrc;
  ssrc.push_back({"s0"});
  msl::Block sbody;
  const msl::SmallVec<msl::Str, 8> sres =
      emitScan(c, sbody, sp, 1, ssrc, snm, adder)[0];

  os << "kernel void agpu_probe_intreduce(device int *out [[buffer(0)]],\n"
        "                                 uint3 tid "
        "[[thread_position_in_threadgroup]]) {\n"
        "  threadgroup int pscr0["
     << rp.scratch.slotsPerOperand
     << "];\n"
        "  int lane = tid.x & 31;\n"
        "  int warp = tid.x / 32;\n"
        "  int r0 = out[tid.x];\n"
        "  int r1 = out[tid.x + 1];\n"
        "  int s0 = r0;\n";
  msl::Printer p(os);
  p.printBlock(rbody);
  p.printBlock(sbody);
  os << "  out[tid.x] = acc0_0";
  for (const msl::Str &r : sres)
    os << " + " << r;
  os << ";\n}\n";
}

// An unstructured region: a loop whose back edge swaps two block parameters,
// the shape a structured lowering cannot express.
void printRegion(std::ostream &os) {
  msl::Context c;

  RegionFacts f;
  f.blocks.resize(3);
  f.blocks[0].defines = {10, 11};
  f.blocks[0].term = TermKind::Branch;
  f.blocks[0].edges = {Edge{1, {10, 11}}};
  f.blocks[1].params = {20, 21};
  f.blocks[1].reads = {20, 21};
  f.blocks[1].term = TermKind::CondBranch;
  f.blocks[1].edges = {Edge{1, {21, 20}}, Edge{2, {}}};
  f.blocks[2].reads = {20};
  f.blocks[2].term = TermKind::Return;

  const RegionPlan p = planRegion(f);

  RegionNames nm;
  nm.namesOf = [](ValueId v) { return ValueNames{"r" + std::to_string(v)}; };
  nm.typeOf = [](ValueId) { return msl::Type::scalar(msl::Scalar::I32); };

  msl::Block body;
  emitRegion(
      c, body, f, p, nm,
      [&](BlockId b) {
        msl::Block inner;
        if (b == 0) {
          inner.push_back(
              c.declStmt(msl::Type::scalar(msl::Scalar::I32), "r10", c.lit(0)));
          inner.push_back(
              c.declStmt(msl::Type::scalar(msl::Scalar::I32), "r11", c.lit(1)));
        }
        if (b == 1)
          inner.push_back(c.assign(
              c.var("r20"), c.binary(msl::BinOp::Add, c.var("r20"), c.lit(1))));
        return inner;
      },
      [&](BlockId) {
        return c.binary(msl::BinOp::Lt, c.var("r20"), c.lit(10));
      });

  os << "kernel void agpu_probe_region(device int *out [[buffer(0)]],\n"
        "                              uint t [[thread_position_in_grid]]) {\n";
  msl::Printer p2(os);
  p2.printBlock(body);
  os << "  out[t] = r20 + r21;\n}\n";
}

// MSL has no forward reference, so the prototype must precede the call and
// the struct must precede both.
void printDeviceFnModule(std::ostream &os) {
  msl::Context c;

  DeviceFnFacts f;
  f.name = "agpu_probe_callee";
  f.params = {DeviceValue{f32(), true, 1}, DeviceValue{i32(), false, 1}};
  f.results = {DeviceValue{f32(), false, 2}};
  f.moduleNeedsPool = true;
  const DeviceFnAbi abi = planDeviceFn(f);

  msl::Printer p(os);

  {
    msl::Block b{emitRetStruct(c, f, abi)};
    p.printBlock(b);
    os << "\n";
  }
  {
    msl::Block b{emitDeviceProto(c, f, abi)};
    p.printBlock(b);
    os << "\n";
  }

  {
    msl::Block body;
    body.push_back(c.declStmt(msl::Type::scalar(msl::Scalar::F32), "r0",
                              c.subscript(c.var("src"), c.var("n"))));
    body.push_back(
        c.declStmt(msl::Type::scalar(msl::Scalar::F32), "r1",
                   c.binary(msl::BinOp::Mul, c.var("r0"), c.litF(2.0))));
    body.push_back(emitDeviceReturn(c, abi, {"r0", "r1"}));

    msl::Block b{emitDeviceFn(c, f, abi, {"src", "n"}, std::move(body))};
    p.printBlock(b);
  }

  KernelFacts kf;
  kf.name = "agpu_probe_caller";
  kf.args = {{"out", f32(), true}, {"in", f32(), true}, {"n", i32(), false}};
  kf.numWarps = 4;
  kf.poolBytes = 1024;
  KernelNames knm;

  KernelResult kr = emitKernel(
      c, kf,
      [&](msl::Context &cc, bool) {
        msl::Block b;
        const CallerContext caller{knm.threadgroupPos, knm.threadId,
                                   knm.gridSize, knm.pool, knm.assertBuffer};
        emitDeviceCall(cc, b, f, abi, {"in", "n"}, caller, {"v0", "v1"});
        b.push_back(
            cc.assign(cc.subscript(cc.var("out"), cc.var(knm.laneId)),
                      cc.binary(msl::BinOp::Add, cc.var("v0"), cc.var("v1"))));
        return b;
      },
      knm);

  msl::Block kb{kr.fn};
  p.printBlock(kb);
}

} // namespace

void printEpilogue(std::ostream &os) {
  msl::Context c;

  const std::vector<EpilogueStep> steps = {
      {"arith.mulf", c.var("alpha")},
      {"math.absf", nullptr},
      {"arith.addf", c.var("bias")},
      {"math.floor", nullptr},
  };
  msl::Expr *chained = epilogueChain(c, steps, c.var("acc"));
  if (!chained)
    return;

  os << "kernel void agpu_probe_epilogue(device float *out [[buffer(0)]],\n"
        "                                uint t [[thread_position_in_grid]])"
        " {\n"
        "  float acc = out[t];\n"
        "  float alpha = 2.0f;\n"
        "  float bias = 1.0f;\n"
        "  out[t] = ";
  msl::Printer p(os);
  p.printExpr(chained);
  os << ";\n}\n";
}

void printMemDesc(std::ostream &os) {
  msl::Context c;
  msl::Block body;

  const MemDesc all = allocMultiBuffered("mdpool", 2, {8, 8});
  const MemDesc one = all.index(1);
  const MemDesc corner = one.subslice({2, 2}, {4, 4});

  body.push_back(memDescDecl(c, all, msl::Type::scalar(msl::Scalar::F32)));

  os << "kernel void agpu_probe_memdesc(device float *out [[buffer(0)]],\n"
        "                               uint t [[thread_position_in_grid]])"
        " {\n"
        "  uint r = t & 3u;\n"
        "  uint k = (t >> 2) & 3u;\n";
  msl::Printer p(os);
  p.printBlock(body);

  os << "  ";
  p.printExpr(memDescElem(c, one, {0, 0}));
  os << " = (float)t;\n  ";
  p.printExpr(memDescElemAt(c, corner, {c.var("r"), c.var("k")}));
  os << " = (float)t;\n"
        "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  out[t] = ";
  p.printExpr(memDescElem(c, corner, {3, 3}));
  os << ";\n}\n";
}

void printBindVecadd(std::ostream &os) {
  Emitter e;
  KernelFacts f;
  f.name = "agpu_probe_bind_vecadd";
  f.args = {{"out", f32(), true}, {"a", f32(), true}, {"b", f32(), true}};

  const LayoutBasis lb{/*reg=*/{32}, /*lane=*/{1, 2, 4, 8, 16},
                       /*warp=*/{}, /*block=*/{}};

  e.addKernel(f, [&](msl::Context &c, bool) {
    msl::Block body;
    SymbolTable sym;
    int t = 0;
    auto fresh = [&] { return "v" + std::to_string(t++); };

    ValueNames idx;
    for (int r = 0; r < 2; ++r) {
      const msl::Str n = fresh();
      body.push_back(c.declStmt(mslTypeOf(i32()), n,
                                rangeElem(c, lb, r, 0, "lane", "warp")));
      idx.push_back(n);
    }
    sym.bindRegs(1, idx);

    ValueId next = 2;
    for (const char *buf : {"a", "b"}) {
      ValueNames ns;
      for (int r = 0; r < 2; ++r) {
        const msl::Str n = fresh();
        body.push_back(c.declStmt(
            mslTypeOf(f32()), n,
            c.subscript(c.var(buf), c.var(*sym.regAt(1, (std::size_t)r)))));
        ns.push_back(n);
      }
      sym.bindRegs(next++, ns);
    }

    ValueNames sum;
    for (int r = 0; r < 2; ++r) {
      const msl::Str n = fresh();
      body.push_back(emitEw(c, EwOp::Add, f32(), n,
                            c.var(*sym.regAt(2, (std::size_t)r)),
                            c.var(*sym.regAt(3, (std::size_t)r))));
      sum.push_back(n);
    }
    sym.bindRegs(4, sum);

    for (int r = 0; r < 2; ++r)
      body.push_back(c.assign(
          c.subscript(c.var("out"), c.var(*sym.regAt(1, (std::size_t)r))),
          c.var(*sym.regAt(4, (std::size_t)r))));
    return body;
  });

  std::ostringstream own;
  e.print(own);
  const std::string text = own.str();
  const std::size_t at = text.find("kernel void");
  if (at != std::string::npos)
    os << text.substr(at);
}

void printCas(std::ostream &os) {
  msl::Context c;

  auto facts = [](ElemClass e, unsigned bits, bool uniform) {
    CasFacts f;
    f.elem = e;
    f.bits = bits;
    f.uniformPtr = uniform;
    return f;
  };

  CasNames inm;
  msl::Block ibody;
  emitCas(c, ibody, planCas(facts(ElemClass::Int, 32, false)), inm, i32());

  CasNames fnm;
  fnm.expected = "fcmp";
  fnm.desired = "fval";
  fnm.result = "fold";
  msl::Block fbody;
  emitCas(c, fbody, planCas(facts(ElemClass::Float, 32, false)), fnm, f32());

  CasNames pnm;
  pnm.expected = "pcmp";
  pnm.desired = "pval";
  pnm.result = "pold";
  msl::Block pbody;
  emitCas(c, pbody, planCas(facts(ElemClass::Int, 16, false)), pnm,
          ElemType{ElemType::Kind::Int, 16, true});

  CasNames unm;
  unm.expected = "ucmp";
  unm.desired = "uval";
  unm.result = "uold";
  unm.shared = "ucasb";
  msl::Block ubody;
  emitCas(c, ubody, planCas(facts(ElemClass::Int, 32, true)), unm, i32());

  os << "kernel void agpu_probe_cas(device atomic_uint *casp [[buffer(0)]],\n"
        "                           device int *out [[buffer(1)]],\n"
        "                           uint3 tid [[thread_position_in_"
        "threadgroup]]) {\n"
        "  int cmp = 1, val = 2, ucmp = 5, uval = 6;\n"
        "  float fcmp = 1.0f, fval = 2.0f;\n"
        "  ushort pcmp = 3, pval = 4;\n"
        "  bool hi = (tid.x & 1) != 0;\n";
  msl::Printer p(os);
  p.printBlock(ibody);
  p.printBlock(fbody);
  p.printBlock(pbody);
  p.printBlock(ubody);
  os << "  out[tid.x] = old + (int)fold + (int)pold + uold_b;\n}\n";
}

void printPoll(std::ostream &os) {
  msl::Context c;
  PollNames nm;

  msl::Block spinning;
  emitPoll(c, spinning, planPoll([] {
             PollFacts f;
             f.bits = 32;
             f.acquire = true;
             return f;
           }()),
           nm);

  PollNames tnm;
  tnm.result = "ready2";
  tnm.flag = "seen2";
  tnm.expected = "want2";
  msl::Block timed;
  emitPoll(c, timed, planPoll([] {
             PollFacts f;
             f.bits = 32;
             f.hasTimeout = true;
             return f;
           }()),
           tnm);

  os << "kernel void agpu_probe_poll(device atomic_uint *flagp [[buffer(0)]],\n"
        "                            device int *out [[buffer(1)]],\n"
        "                            uint3 tid [[thread_position_in_"
        "threadgroup]]) {\n"
        "  uint want = 1;\n"
        "  uint want2 = 2;\n";
  msl::Printer p(os);
  p.printBlock(spinning);
  p.printBlock(timed);
  os << "  out[tid.x] = (int)ready + (int)ready2;\n}\n";
}

void printCombiners(std::ostream &os) {
  const ElemType types[] = {f32(),
                            f16(),
                            i32(),
                            ElemType{ElemType::Kind::Int, 32, true},
                            ElemType{ElemType::Kind::Int, 16},
                            ElemType{ElemType::Kind::Int, 16, true}};
  os << "kernel void agpu_probe_combiners(device float *out [[buffer(0)]],\n"
        "                                 uint3 tid "
        "[[thread_position_in_threadgroup]]) {\n"
        "  float sink = 0;\n";
  int n = 0;
  for (unsigned i = 1; i < unsigned(Combiner::Count); ++i) {
    const Combiner fn = Combiner(i);
    for (const ElemType &e : types) {
      const char *r = simdReduceFn(fn, e);
      const char *pi = simdPrefixInclusiveFn(fn, e);
      const char *px = simdPrefixExclusiveFn(fn, e);
      if (!r && !pi && !px)
        continue;
      const std::string v = "cv" + std::to_string(n++);
      std::ostringstream ty;
      msl::Printer(ty).printType(mslTypeOf(e));
      os << "  " << ty.str() << " " << v << " = (" << ty.str() << ")tid.x;\n";
      for (const char *fnName : {r, pi, px})
        if (fnName)
          os << "  sink += (float)" << fnName << "(" << v << ");\n";
    }
  }
  os << "  out[tid.x] = sink;\n}\n";
}

void printPlannedModule(std::ostream &os) {
  Emitter e;

  KernelFacts f;
  f.name = "agpu_probe_module";
  f.args = {{"out", f32(), true}, {"n", i32(), false}};
  f.numWarps = 4;

  e.addKernel(f, [&](msl::Context &c, bool) {
    msl::Block b;
    b.push_back(c.assign(c.subscript(c.var("out"), c.var("lane")),
                         c.call(helperName(Helper::Erf), {c.litF(0.5)})));
    return b;
  });
  e.helpers.add(Helper::Erf);

  std::ostringstream own;
  e.print(own);

  const std::string text = own.str();
  const std::size_t kernelAt = text.find("kernel void");
  if (kernelAt != std::string::npos)
    os << text.substr(kernelAt);
}

int main() {
  HelperSet helpers;
  for (unsigned i = 0; i < unsigned(Helper::Count); ++i)
    helpers.add(Helper(i));

  printPrelude(std::cout, helpers);
  printHelperUses(std::cout);
  std::cout << "\n";

  HelperSet kernelHelpers;
  printEmittedKernel(std::cout, kernelHelpers);
  std::cout << "\n";

  printRaggedPanel(std::cout);
  std::cout << "\n";

  printDeviceFnModule(std::cout);
  std::cout << "\n";

  printElementwise(std::cout);
  std::cout << "\n";

  printRegion(std::cout);
  std::cout << "\n";

  printIntegerReduce(std::cout);
  std::cout << "\n";

  printLoops(std::cout);
  std::cout << "\n";

  printPlannedDot(std::cout);
  std::cout << "\n";

  printPoll(std::cout);
  std::cout << "\n";

  printCas(std::cout);
  std::cout << "\n";

  printEpilogue(std::cout);
  std::cout << "\n";

  printMemDesc(std::cout);
  std::cout << "\n";

  printCombiners(std::cout);
  std::cout << "\n";

  printPlannedModule(std::cout);
  std::cout << "\n";

  printBindVecadd(std::cout);
  return 0;
}
