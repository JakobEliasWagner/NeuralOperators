FROM python:3.11.5-slim-bookworm as base

ENV  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  # Poetry's configuration:
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local' \
  POETRY_VERSION=1.7.1

RUN apt-get -y update; apt-get -y install curl git

RUN curl -sSL https://install.python-poetry.org | python3 -

FROM base as dependencies

WORKDIR /nos

COPY poetry.lock pyproject.toml /nos/
COPY continuity /nos/continuity/

RUN poetry install  --no-interaction --no-ansi --no-root

FROM dependencies as nos

WORKDIR /nos

COPY . /nos/

RUN poetry install


FROM base as nos-test

WORKDIR /nos

COPY . /nos/

RUN poetry install --with=dev,test
