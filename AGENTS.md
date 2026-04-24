# `AGENTS.md`

## Environment

- First, read the `README.md` [prerequisites](./README.md#prerequisites) to
  ensure all dependencies are installed.

- Then setup a Python virtual environment and install dependencies:

  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

- Next, retrieve the latest Triton and LLVM binaries for building extensions:

  ```bash
  python ci/download-artifact.py llvm
  python ci/download-artifact.py triton
  ```

  If the user wants a specific version, read the documentation of
  `ci/download-artifact.py` to specify the commit, OS, and architecture; e.g.,
  `python ci/download-artifact.py llvm-abc1234-linux-x64`. Note that available
  artifacts are limited; to list available artifacts, run
  `ci/fetch-artifacts.py`.

## Build

With the environment in place, build the extensions:

```bash
make build
```

Read the [`README.md`](./README.md#build) for more details on the build process.

## Test

Run the test suite to verify the extensions are working correctly (also see the
[`README.md`](./README.md#test)):

```bash
make test
```

Do not commit changes that cause test failures. Do not disable tests. To add new
kinds of test runners (e.g., pyunit), append commands to the `make test` target.

## Commit

- Before adding new functionality, read the
  [contributing guidelines](./CONTRIBUTING.md) and align plans to that.

- Always run the pre-commit hooks and fix any issues before committing changes:

  ```bash
  pre-commit run --all-files
  ```

- Disclose AI assistance with an `Assisted-by` trailer naming your agent and
  model:
  `git commit -s -m "Add support for X" --trailer "Assisted-by: <agent-name>/<model-id>"`.

- Open a PR with a clear description of the changes and link any relevant issues
  or discussions.
