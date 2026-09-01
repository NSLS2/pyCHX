Contributing
============

Contributions are welcome. Please use a focused branch and pull request for
each change.

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/NSLS2/pyCHX/issues. Include relevant details
about your environment and the steps needed to reproduce the problem.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for proposed features. Keep changes focused and
describe any user-visible behavior in the pull request.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Documentation improvements are welcome in the Sphinx documentation, README,
and source docstrings.

Local Development
-----------------

1. Fork the ``pyCHX`` repository on GitHub.
2. Clone your fork locally::

    $ git clone https://github.com/YOUR_USERNAME/pyCHX.git
    $ cd pyCHX

3. Create and activate a virtual environment::

    $ python -m venv .venv
    $ source .venv/bin/activate

4. Install the runtime and development dependencies::

    $ python -m pip install --group facility-source
    $ python -m pip install -e ".[test,docs,facility]"

5. Create a branch for local development::

    $ git switch -c name-of-your-bugfix-or-feature

6. Run the configured formatting checks and tests::

    $ pre-commit run --all-files
    $ pytest

7. Commit your changes, push your branch, and submit a pull request to the
   ``NSLS2/pyCHX`` ``main`` branch.

Pull Request Guidelines
-----------------------

* Include tests for behavioral changes.
* Update relevant documentation and docstrings for new functionality.
* Ensure the configured pre-commit and GitHub Actions checks pass.
