# Testing

Extension tests run via `pytest`. Tests can be:

- either **normal `pytest` tests** (e.g. in `test/` subdirectories), or
- **`.mlir` (lit-style) tests**.

## `.mlir` (lit-style) tests

For this style of test, each `.mlir` file keeps a `// RUN:` directive and its
`// CHECK` lines, but the `RUN` line now invokes a small per-extension *driver
script* and pipes the result into the Python [`filecheck`][filecheck] package:

```mlir
// RUN: loop_split.py %s | %filecheck %s
```

This requires:

- A **driver script** (e.g. [`loop_split.py`][loop_split]) imports the extension
  (registering its passes/dialects) and calls [`run_passes`][mlir_runner] with
  the pass pipeline to apply.

- The **pass runner**, [`mlir_runner.py`][mlir_runner] reads an MLIR file, runs
  the passes, and prints the result to stdout.

- The top-level [`conftest.py`][conftest] **collects** every `.mlir` file as a
  pytest item, substitutes `%s` / `%filecheck` / the driver script, and runs the
  `RUN` line as a shell pipeline. The following substitutions apply:

  | token        | expands to                                        |
  | ------------ | ------------------------------------------------- |
  | `%s`         | the `.mlir` file path                             |
  | `%filecheck` | `python -m filecheck --enable-var-scope`          |
  | `<name>.py`  | `python <testdir>/<name>.py` (if it exists there) |

## Run

To run all tests:

```bash
pytest
```

To run tests for a single extension:

```bash
cd pass/LoopSplit && pytest. # or...
pytest pass/LoopSplit
```

[conftest]: ../confttest.py
[filecheck]: https://pypi.org/project/filecheck/
[loop_split]: pass/LoopSplit/test/loop_split.py
[mlir_runner]: mlir_runner.py
