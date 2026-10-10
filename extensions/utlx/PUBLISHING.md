# Publishing `triton-utlx` to PyPI

The `triton-utlx` wheel is built and uploaded by the
[`publish-utlx.yml`](../../.github/workflows/publish-utlx.yml) GitHub Actions
workflow. Nobody uploads wheels by hand: the workflow builds the wheel against
the pinned Triton release, checks it, smoke-tests it against stock PyPI Triton
and publishes it with PyPI Trusted Publishing (no API token involved).

## TL;DR

```bash
# 1. Dry run: build and test only, publish nothing
gh workflow run publish-utlx.yml -R triton-lang/triton-ext \
  -f version=3.8.0.post12 -f publish_to=none

# 2. Optional: rehearse the upload on TestPyPI
gh workflow run publish-utlx.yml -R triton-lang/triton-ext \
  -f version=3.8.0.post12 -f publish_to=testpypi

# 3. Publish for real
gh workflow run publish-utlx.yml -R triton-lang/triton-ext \
  -f version=3.8.0.post12 -f publish_to=pypi
```

Pick the next unused version first (see
[Choosing a version](#choosing-a-version)). PyPI never accepts the same version
twice.

## Prerequisites

- The [`gh` CLI](https://cli.github.com/), logged in (`gh auth login`) as an
  account with write access to `triton-lang/triton-ext`. Triggering a
  `workflow_dispatch` workflow requires write access.
- The changes you want to ship are on the branch the workflow will build,
  normally `main` (see [Which code gets built](#which-code-gets-built)).
- One-time setup, already done for `triton-utlx`: a project owner registered a
  Trusted Publisher at
  <https://pypi.org/manage/project/triton-utlx/settings/publishing/> with
  repository `triton-lang/triton-ext`, workflow `publish-utlx.yml` and
  environment `pypi`, and the same on test.pypi.org with environment `testpypi`.
  If the upload step fails with an OIDC / "invalid publisher" error, this
  registration is what to check.

## Workflow inputs

| Input        | Values                               | Meaning                                        |
| ------------ | ------------------------------------ | ---------------------------------------------- |
| `version`    | e.g. `3.8.0.post12`                  | Version written into the wheel. Required.      |
| `publish_to` | `none` (default), `testpypi`, `pypi` | Where to upload. `none` builds and tests only. |

## Choosing a version

The version must be the Triton release `extensions/utlx/pyproject.toml` pins
(`dependencies = ["triton~=3.8.0"]`), optionally with a `.postN` suffix:
`3.8.0`, `3.8.0.post1`, `3.8.0.post2`, ... The workflow rejects anything else,
because the wheel's `triton~=` pin is only truthful if its version tracks the
Triton release it was built against.

Find the latest published version and take the next `.postN`:

```bash
pip index versions triton-utlx
# triton-utlx (3.8.0.post11)
# Available versions: 3.8.0.post11, 3.8.0.post10, ...
```

You do **not** need to edit `version = ...` in `pyproject.toml`; the workflow
overwrites it with the `version` input before building.

PyPI never accepts a version twice, even after the release is deleted or yanked.
If a published wheel is bad, fix it and publish the next `.postN`.

## Which code gets built

`gh workflow run` builds the workflow's ref, which is the repository's default
branch (`main`) unless you pass `--ref`:

```bash
gh workflow run publish-utlx.yml -R triton-lang/triton-ext --ref my-branch \
  -f version=3.8.0.post12 -f publish_to=none
```

Use `--ref` for dry runs of unmerged work. Publish to PyPI from `main`, so every
released version corresponds to merged code.

## What the workflow does

The `build` job runs in the `quay.io/pypa/manylinux_2_28_x86_64` image (the one
Triton's own wheels use) and takes a while, because it builds Triton from
source; the job timeout is 5 hours.

1. **Resolve versions**: reads the `triton~=X.Y.Z` pin from `pyproject.toml` and
   checks that `version` is `X.Y.Z` or `X.Y.Z.postN`. It also checks that the
   first line of `PYPI_README.md` (the PyPI project description) names
   `Triton X.Y build`.
1. **Build Triton `vX.Y.Z` with extension headers**: PyPI Triton wheels ship no
   C++ headers, so the pinned release is built from source with
   `TRITON_EXT_ENABLED=1` and its headers are installed into the tree.
1. **Build the wheel** with clang (as PyPI Triton and its LLVM are built; a
   GCC-built plugin disagrees with PyPI's `libtriton.so` about MLIR trait
   TypeIDs), then retags it `py3-none-manylinux_2_28_x86_64`.
1. **Check glibc and libstdc++ symbol versions**: every versioned symbol
   `libutlx.so` needs must exist in the image's system libraries, or the
   `manylinux_2_28` tag would be wrong.
1. **Smoke test against PyPI Triton**: installs the wheel into a fresh venv,
   imports `utlx_plugin` and `tlx`, checks `TRITON_PLUGIN_PATHS`, checks that
   every symbol `libutlx.so` needs from `libtriton.so` is exported, and loads
   the library.
1. **Upload wheel artifact** `triton-utlx-wheel`.

If `publish_to` is not `none`, the `publish` job then downloads that artifact
and uploads it to PyPI or TestPyPI through the GitHub environment of the same
name. If that environment has protection rules (required reviewers), approve the
deployment in the run's page on GitHub.

## Following a run

```bash
gh run list -R triton-lang/triton-ext -w publish-utlx.yml -L 5
gh run watch -R triton-lang/triton-ext <run-id>
gh run view -R triton-lang/triton-ext <run-id> --log-failed
```

To test a dry-run wheel locally before publishing, download the artifact:

```bash
gh run download -R triton-lang/triton-ext <run-id> -n triton-utlx-wheel -D dist
pip install --force-reinstall --no-deps dist/triton_utlx-*.whl
```

## Verifying the release

```bash
python -m venv /tmp/utlx-check && source /tmp/utlx-check/bin/activate
pip install triton-utlx==3.8.0.post12
python -c "import importlib.metadata as md, triton, utlx_plugin; \
print(triton.__version__, md.version('triton-utlx'))"
```

A TestPyPI release installs with the extra index supplying Triton, which is not
on TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ triton-utlx==3.8.0.post12
```

The project page is <https://pypi.org/p/triton-utlx>
(<https://test.pypi.org/p/triton-utlx> for TestPyPI). It can take a minute
before `pip` sees a new version.

## Troubleshooting

| Failure                                                       | Cause and fix                                                                                                        |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `version ... must be X.Y.Z or X.Y.Z.postN`                    | The version doesn't match the `triton~=` pin. Fix the input.                                                         |
| `PYPI_README.md heading must name "Triton X.Y build"`         | The pin moved to a new Triton minor; update the first line of `PYPI_README.md`.                                      |
| `File already exists` on upload                               | That version is already on the index. Use the next `.postN`.                                                         |
| Build succeeded, publish failed (e.g. transient upload error) | Rerun only the failed job: `gh run rerun -R triton-lang/triton-ext <run-id> --failed`. The wheel artifact is reused. |
| `libutlx.so needs GLIBC_...`                                  | Something linked against a newer glibc/libstdc++ than manylinux_2_28 allows.                                         |
| `symbols missing from libtriton`                              | The plugin uses a Triton symbol that PyPI's `libtriton.so` doesn't export. The plugin must not depend on it.         |

## Moving to a new Triton release

When the plugin moves to a new Triton (say 3.9.0):

1. Update `dependencies = ["triton~=3.9.0"]` in `pyproject.toml`.
1. Update the first line of `PYPI_README.md` to say `Triton 3.9 build`.
1. Merge, then publish `3.9.0` (or `3.9.0.post1`) with the commands above.
