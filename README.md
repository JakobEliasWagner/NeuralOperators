<div align="center">
<img alt="Neural Operators" src="doc/logo.png" width=30%>
<h1>Neural Operators</h1>

Learning neural operators for parameterized geometries in the context of sonic crystals and the acoustic Helmholtz equation.
The data-generation of this problem has been moved to a dedicated
[repository](https://github.com/JakobEliasWagner/Helmholtz-Sonic-Crystals) as this implementation does not directly
touch the implementation of the operators.

[![Python 3.8](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![Code Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JakobEliasWagner/715271f51dd7b16c37fcf84c79dcb31a/raw/covbadge.json)](https://jakobeliaswagner.github.io/Neural-Operators/_static/codecov/index.html)
[![Documentation](https://img.shields.io/badge/Documentation-FF7043)](https://jakobeliaswagner.github.io/Neural-Operators/index.html)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/jakob-wagner-65b9871a9/)
</div>

## Setup

To install all required and optional dependencies run

```shell
poetry install --with=optimize,test,dev,doc
```
- **optimize**: adds dependencies for optimizing models and probe the training of operators.
- **test**: adds `pytest` and `pytest-cov ` for coverage reports and tests,
- **dev**: to contribute and to ensure code quality,
- **doc**: installs required modules to build the documentation locally.

## Hooks

Ensure the `dev` optional dependencies are installed.
Install the git hook scripts

```shell
pre-commit install
```

now `pre-commit` will run automatically on `git commit `.

## Tests

Ensure the `test` optional dependencies are installed.
Run tests with

```shell
pytest test/
```

## Documentation

The documentation of this project can be read [here](https://jakobeliaswagner.github.io/Neural-Operators/index.html).
Ensure the `doc` optional dependency group is installed.

To build the documentation locally run

```shell
pytest --cov=src/nos --cov-report html:doc/source/_static/codecov test/
sphinx-apidoc -f -o doc/source/ src/nos
sphinx-build -M html doc/source doc/build
```

These commands build a coverage report for the project, automatically documents the project using docstrings and builds
the documentation.
The build documentation can then be found in `doc/build/html`. Open the `index.html` to access it.
