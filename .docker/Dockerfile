FROM python:3.11-buster as base

RUN pip install poetry==1.7.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /home

COPY . ./

RUN poetry install --with=doc,test,optimize --no-root --no-cache
RUN poetry cache clear --all

FROM python:3.11-buster as base-cuda

ENV VIRTUAL_ENV=/home/.venv \
    PATH="/home/.venv/bin:$PATH"

COPY --from=base ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /home

RUN source .venv/bin/activate
RUN pip install torch==2.1.0
RUN pip cache purge

FROM python:3.11-buster as nos

RUN pip install poetry==1.7.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY --from=base-cuda ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /home

COPY . ./
RUN poetry install --with=doc,test,optimize --no-cache

RUN poetry cache clear --all

FROM python:3.11-slim-buster as runtime

ENV VIRTUAL_ENV=/home/.venv \
    PATH="/home/.venv/bin:$PATH"

COPY --from=nos ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /home

COPY . ./

RUN pre-commit install
