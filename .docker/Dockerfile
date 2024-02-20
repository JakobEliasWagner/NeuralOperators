FROM python:3.11.5-slim-bookworm as base

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random

WORKDIR /home

FROM base as builder

ENV PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_NO_INTERACTION=1 \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local' \
  POETRY_VERSION=1.7.1

RUN apt-get update && apt-get install curl -yy

RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}

FROM builder as dependencies1

COPY . /home/

RUN poetry install --no-interaction --no-ansi --no-root
RUN rm -rf $POETRY_CACHE_DIR

FROM base as dependencies2

COPY --from=dependencies1 /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN pip install torch  # to build the correct dependent packages (poetry currently does not build them)

FROM dependencies as nos

COPY --from=dependencies2 /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN poetry install --no-interaction --no-ansi

FROM nos as dev

RUN poetry install --no-interaction --no-ansi --with=optimize,doc,dev
RUN pre-commit install
