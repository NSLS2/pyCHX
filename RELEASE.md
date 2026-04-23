Steps for releasing a version of pyCHX

- Make sure that you have an up-to-date copy of the master branch.
- Make an empty commit to serve as a marker in the history. This is technically
  optional, but it is nice to do. `git commit --allow-empty -m "REL: v0.0.1"`
- Now make the tag. This should never be delete, so make sure you are certain.
  `git tag v0.0.1`
- Push the commit up to github: `git push upstream master`
- Push the tag also: `git push upstream v0.0.1`
