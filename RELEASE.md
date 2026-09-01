# Releasing pyCHX

PyPI publication is handled by `.github/workflows/python-publish.yml` when a
GitHub release is published.

1. Update the local `main` branch and run the tests, pre-commit hooks,
   documentation build, and distribution checks.
2. Build the wheel and source distribution and run `twine check dist/*`.
3. Install the wheel in a clean environment and verify its imports.
4. Create an annotated version tag, for example
   `git tag -a v0.0.1 -m "REL: v0.0.1"`.
5. Push `main` and the tag to the upstream repository.
6. Create and publish the corresponding GitHub release.
7. Verify the PyPI publication workflow completes successfully.
