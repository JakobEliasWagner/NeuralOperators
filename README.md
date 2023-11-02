# Neural-Operators
*Master's Thesis by Jakob Wagner*

[![Python 3.8](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
![Code Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JakobEliasWagner/715271f51dd7b16c37fcf84c79dcb31a/raw/e1ecebc4ebfaac81fe3225be6d84ebe1069231c2/covbadge.json)
[![Code Coverage Detailed](https://img.shields.io/badge/Detailed%20coverage%20report-008000)](https://jakobeliaswagner.github.io/Neural-Operators/_static/codecov/index.html)
[![Documentation](https://img.shields.io/badge/Documentation-FF7043)](https://jakobeliaswagner.github.io/Neural-Operators/index.html)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/jakob-wagner-65b9871a9/)

## Setup
I recommend setting up a python venv.

Install the required packages and libraries
```shell
pip install -r requirements.txt
```

## Hooks
Install the pre-commit package manager:
```shell
pip install pre-commit
```
Install the git hook scripts
```shell
pre-commit install
```
now `pre-commit` will run automatically on `git commit `.

## Tests
Run tests
```shell
pytest test/
```
