# Porting TLX TDM ops to triton-ext (draft)

## Summary

This draft adds TLX spellings of the AMD CDNA5 Tensor Data Mover (TDM) ops to
uTLX, following the repository's portable-op convention. The ops originate in
upstream Triton (`triton-lang/triton`), where the Gluon frontend exposes them as
`triton.experimental.gluon.language.amd.cdna5.tdm`. Nothing in TLX could name
them before: `utlx_plugin` had NVIDIA TMA descriptor ops
(`async_descriptor_load` and friends) but no AMD descriptor data-movement ops at
all.

Eight language-visible upstream builtins are ported. Each gets a `tlx.tdm_*`
Python builtin in `mem_ops.py`, a portable C++ wrapper in `ops/NewOps.cpp`, a
declaration in `ops/NewOps.h`, and an entry in the plugin op table in
`uTLXPlugin.cpp`:

| Upstream Gluon builtin         | TLX builtin                        | MLIR op created                             |
| ------------------------------ | ---------------------------------- | ------------------------------------------- |
| `tdm.async_load`               | `tlx.tdm_async_load`               | `amdg.async_tdm_copy_global_to_local`       |
| `tdm.async_load_fused`         | `tlx.tdm_async_load_fused`         | `amdg.async_tdm_fused_copy_global_to_local` |
| `tdm.async_store`              | `tlx.tdm_async_store`              | `amdg.async_tdm_copy_local_to_global`       |
| `tdm.async_gather`             | `tlx.tdm_async_gather`             | `amdg.async_tdm_gather`                     |
| `tdm.async_scatter`            | `tlx.tdm_async_scatter`            | `amdg.async_tdm_scatter`                    |
| `tdm.prefetch`                 | `tlx.tdm_prefetch`                 | `amdg.tdm_prefetch`                         |
| `tdm.async_wait`               | `tlx.tdm_async_wait`               | `amdg.async_tdm_wait`                       |
| `tdm.update_tensor_descriptor` | `tlx.tdm_update_tensor_descriptor` | `amdg.update_tensor_descriptor`             |

`amdg.async_tdm_intrinsic_wait` is deliberately not ported: upstream creates it
only inside the AMD lowering (it counts emitted intrinsics, not IR ops), never
from the language.

Descriptors are the ones TLX already makes with `tlx.make_tensor_descriptor`;
sources and destinations are `tlx.buffered_tensor` shared-memory buffers,
exactly as with `async_descriptor_load`/`async_descriptor_store`. As upstream, a
descriptor must be positioned with `tdm_update_tensor_descriptor` (or the
`offsets=` convenience on the load/store builtins, which emits that update
first) before a copy reads it.

## What "current status" means here

Two upstream revisions were compared:

- the Triton pinned by this repository (`ci/triton-hash.txt`, `0cf3bf6e`, the
  3.9.0 nightly line), and
- latest `main` of `triton-lang/triton` at porting time (`c778adf`).

The core TDM ops already exist at the pin with the same MLIR names, so the
wrappers work against the pinned wheel today. Everything upstream added after
the pin is also carried, so the port does not go stale at the next pin bump:

- `scope=` (`"cta"`/`"gpu"`/`"sys"`) and `cache_policy=` (gfx12 temporal hint:
  `regular`/`none`/`high`/`last_use`) on all five copy ops, with the legacy
  `cache_modifier=` deprecated upstream in favour of them. TLX accepts
  `cache_policy` as the temporal-hint string rather than upstream's
  `CachePolicy` object, because plugin ops can only pass values (see below); the
  strings are the same.
- The fused load (`async_load_fused`, 2-4 members) and
  `update_tensor_descriptor` with `add_offsets`/`set_bounds`/`pred`/
  `clamp_bounds`, both already present at the pin and included here so the TLX
  surface matches upstream's `__all__` one for one.

## The reference implementation

The port mirrors one existing portable op end to end: `utlx_async_tma_prefetch`
(with `utlx_read_barrier_phase` as the AMD precedent and `utlx_fence` for enum
encoding). Its anatomy, and where each piece of the TDM port follows it:

1. **C++ wrapper by registered name** (`ops/NewOps.cpp`).
   `createAsyncTMAPrefetch` never names an MLIR op class; it calls the
   file-local `createRuntimeOp` helper, which looks the op up with
   `mlir::RegisteredOperationName::lookup("ttng.async_tma_prefetch")`. The TDM
   wrappers do the same with `amdg.*` names, so this file needs no AMD dialect
   headers and the plugin still builds and loads on a Triton without the AMD
   dialect. A missing op emits
   `uTLX: '<name>' is not registered in this Triton build` and leaves its result
   slot null.
1. **Declaration + op-table entry** (`ops/NewOps.h`, `uTLXPlugin.cpp`). The
   eight `createAsyncTDM*`/`createTDMPrefetch`/ `createUpdateTensorDescriptor`
   functions are declared next to `createReadBarrierPhase` and registered as
   `utlx_async_tdm_*` beside it. The table is sized with `std::size(ops)`, so no
   count to update.
1. **Python builtin** (`mem_ops.py`). Like `async_descriptor_prefetch_tensor`,
   each `tdm_*` builtin is an `@tl.builtin` that validates its arguments,
   converts them to IR handles, calls `builder.utlx_<op>([...])`, and wraps
   value results with `checked_handle` from `_compat.py`, which turns the null
   slot from step 1 into a Python `RuntimeError` at the call site instead of a
   crash in a later pass. The new names are exported from
   `utlx_plugin/__init__.py` (`__all__` plus the `mem_ops` import), so they are
   visible as `tlx.tdm_*` in kernels.
1. **Attributes as trailing i32 constants.** Plugin ops only take
   `mlir::Value`s, so, exactly as `createFence` decodes its scope from an i32
   constant, the TDM wrappers take trailing code operands: `cache` (-1 omit; 0
   none, 1 `.ca`, 2 `.cg`, 3 `.wb`, 4 `.cs`, 5 `.wt`, 6 `.cv`), `scope` (-1
   omit; 0 cta, 1 gpu, 2 sys) and `policy` (-1 omit; 0 regular, 1 none, 2 high,
   3 last_use). The C++ side decodes them with the existing `extractConstInt`
   helper and builds the real attributes: `CacheModifierAttr`/`MemSyncScopeAttr`
   from the Triton dialect (headers already included), and
   `#amdg.cache_policy_gfx12<...>` parsed from its assembly form with
   `mlir::parseAttribute`, silently omitted if the AMD dialect is not loaded. An
   optional mbarrier operand is present only when the caller passes one; the
   wrappers tell from the operand count, as `createAsyncLoad` does for its
   optional mask/other.
1. **Token results.** The copy ops and the wait return `!ttg.async_token`
   upstream; the wrappers create that result type (`ttg::AsyncTokenType`, as
   `createAsyncLoad` does) and the Python builtins return `tlx.async_token`, so
   `tdm_async_wait(0, tokens)` can chain them. `tdm_prefetch` is void, like the
   reference op.

Two operand-protocol details are worth calling out. The fused op and
`update_tensor_descriptor` have variadic operand groups, whose sizes MLIR reads
from an `operandSegmentSizes` attribute that the generic creation path does not
populate (the failure `createAsyncLoad`'s comment describes); both wrappers set
it explicitly. And `update_tensor_descriptor`'s counts-first encoding
(`num_add, add..., num_set, set..., has_pred, [pred], clamp`) keeps the two
variadic groups unambiguous without a rank operand.

## Tests

`test/test_tdm_ops.py` is the focused maintained test, gated on a CDNA5
(gfx1250) target via the new `is_hip_cdna5()` helper in `test/conftest.py`, next
to the existing `is_hip_cdna4()`:

- load -> `local_load` -> store round trip, and a load/store round trip between
  two descriptors;
- `update_tensor_descriptor` + `prefetch` positioning;
- gather/scatter restoring a tile through a reversed row index set;
- a fused two-member load with disjoint warp hints;
- a compile-only check that `scope=`/`cache_policy=` reach the TTGIR as
  `cachePolicy`/`scope` on `amdg.async_tdm_copy_global_to_local`.

## Verification status

Done in this environment:

- `python -m py_compile` on all changed Python files: pass.
- `clang-format --dry-run --Werror` on the changed C++ files (after
  `clang-format -i`): pass; the diff is additions only.
- `pre-commit run yapf` and `ruff` on the changed Python files: pass.

Not done, with reasons:

- **Wheel build.** `make -C extensions/utlx build` (pinned Triton wheel
  `3.9.0+git0cf3bf6e` and LLVM `18f9e623`, both fetched with the `ci/` scripts
  into the project `.venv`) fails in `mlir-tblgen` on
  `tlx/dialect/include/IR/TLXOps.td` (`missing attribute 'buffer_shape'`). The
  identical failure reproduces with all of this draft's changes stashed, so it
  pre-dates the port; the TDM code is downstream of that step and was not
  reached by the compiler here.
- **GPU tests.** `test_tdm_ops.py` was not run: the plugin wheel does not build
  (above), and this machine has no gfx1250 device, so the tests would skip even
  with a wheel. They are written to run under `make -C extensions/utlx test` on
  CDNA5 hardware.

## Draft status and how to take it forward

The work is on the local branch `tdm-portable-ops` as a single commit; nothing
is pushed. To publish it as a draft pull request against
`triton-lang/triton-ext` from this checkout:

```bash
git push fork tdm-portable-ops
gh pr create --draft --repo triton-lang/triton-ext \
  --head njriasan:tdm-portable-ops --base main \
  --title "[uTLX][AMD] Port TLX TDM ops as portable ops" \
  --body-file extensions/utlx/TDM_PORT_REPORT.md
```

(`fork` is `https://github.com/njriasan/triton-ext.git`; `gh` is not installed
on this machine, which is why the push is left to the reader.)

Known follow-ups before this leaves draft:

- Run the new tests on gfx1250 hardware and fix whatever the
  `operandSegmentSizes` property spelling, the parsed cache-policy attribute, or
  the descriptor layout checks object to first.
- Decide whether TLX should also grow upstream's `CachePolicy` object and
  host-side TDM descriptors; this draft keeps the smaller string encoding and
  device-side descriptors only.
- The `tlx/language/tlx` vendored copy is a snapshot of the fork frontend and
  was left untouched; only the shipped `utlx_plugin` package gained the ops.
