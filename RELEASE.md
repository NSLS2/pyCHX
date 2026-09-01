# Releasing pyCHX

Before publishing a release, confirm which PyPI workflow is intended to run;
the repository currently contains two publication workflows.

1. Update the local `main` branch and run the configured tests and checks.
2. Create an annotated version tag, for example
   `git tag -a v0.0.1 -m "REL: v0.0.1"`.
3. Push `main` and the tag to the upstream repository.
4. Create and publish the corresponding GitHub release.
5. Verify the expected PyPI publication workflow completes successfully.
