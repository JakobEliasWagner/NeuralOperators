FROM python:3.11-buster as base

RUN pip install poetry==1.7.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /home

COPY . ./

RUN poetry install --with=doc,test,optimize,cpu,dev --without=gpu --no-root --no-cache

FROM base as nos

RUN poetry install --with=doc,test,optimize,cpu,dev --without=gpu --no-cache

FROM python:3.11-slim-buster as runtime

ENV VIRTUAL_ENV=/home/.venv \
    PATH="/home/.venv/bin:$PATH"

COPY --from=nos ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /home

COPY . ./
