# Contributing

## Extensions

To contribute a new extension, follow these steps:

- decide what directory to add to based on the scope of functionality: dialects
  and their operations go in [`dialect`](./dialect), new passes in
  [`pass`](./pass), etc. If the addition spans multiple top-level directories
  but should not logically be split up, use [`extensions`](./extensions).
- create a new directory for the addition
- add the implementation files, CMakeLists.txt, and `triton-ext.conf`; provide
  tests and ensure they run in `make test`.
- check that all pre-commit checks pass: `pre-commit run --all-files`
- open a PR with a clear description of the extension and its functionality, and
  link any relevant issues or discussions.

Note that this project may reject contributions that do not follow the above
guidelines or contributions with no clear plan for maintenance or deployment.

## Bugs, features, enhancements

Submit an issue to the [issue tracker]. Include a clear description of the
problem or feature request, including steps to reproduce if applicable.

### Reproduce a CI failure

Use the CI artifacts to reproduce a failure locally:

```console
git checkout <commit>
python ci/download-artifact.sh llvm-<commit>-<os>-<arch>
python ci/download-artifact.sh triton-<commit>-<os>-<arch>
<failing command>
```

### Submitting a pull request

Prior to submitting a pull request, ensure that all tests pass and that the code
follows the project's coding standards. This includes running all pre-commit
hooks:

```bash
make test
pre-commit run --from-ref origin/main --to-ref HEAD
```

[issue tracker]: https://github.com/triton-lang/triton/issues
