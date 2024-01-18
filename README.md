# Neural-Operators

*Master's Thesis by Jakob Wagner*

[![Python 3.8](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![Code Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JakobEliasWagner/715271f51dd7b16c37fcf84c79dcb31a/raw/e1ecebc4ebfaac81fe3225be6d84ebe1069231c2/covbadge.json)](https://jakobeliaswagner.github.io/Neural-Operators/_static/codecov/index.html)
[![Documentation](https://img.shields.io/badge/Documentation-FF7043)](https://jakobeliaswagner.github.io/Neural-Operators/index.html)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/jakob-wagner-65b9871a9/)

## Setup

Install the required packages and libraries
```shell
pip install .
```

To additionally install all packages and libraries required for testing

```shell
pip install .[test]
```

To additionally install all packages and libraries required for developing

```shell
pip install .[dev]
```

## Hooks
Ensure the `dev` optional dependencies are installed.
Then install the pre-commit package manager:
```shell
pip install pre-commit
```
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

To install the necessary tools to build the documentation

```shell
pip install sphinx
```

To build the documentation locally run

```shell
sphinx-build -M html doc/source doc/build
```

The build documentation can then be found in `doc/build/html`. Open the `index.html` to access it.
However, the local build of the documentation does not have access to the code-coverage report as this report is
generated by a GitHub pipeline.

To update the documentation on the current API

```shell
sphinx-apidoc -f -o doc/source/ src/nos
```
