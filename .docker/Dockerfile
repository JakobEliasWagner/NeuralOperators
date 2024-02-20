FROM python:3.11.5-slim-bookworm as base

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local' \
  POETRY_VERSION=1.7.1

RUN apt-get update && apt-get install curl git -yy

RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}

FROM base as dependencies

WORKDIR /home
COPY poetry.lock pyproject.toml /home/
COPY continuity /home/continuity

RUN poetry install --no-interaction --no-ansi --no-root
RUN pip install torch  # to build the correct dependend packages (poetry currently does not build them)

COPY . /home/

FROM dependencies as nos

WORKDIR /home

RUN poetry install --no-interaction --no-ansi

FROM nos as dev

WORKDIR /home

RUN poetry install --no-interaction --no-ansi --with=optimize,doc,dev
RUN pre-commit install
