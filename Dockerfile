ARG PYTHON_BASE=3.13-slim
# build stage
FROM python:$PYTHON_BASE AS builder

# install PDM
RUN pip install -U pdm
# disable update check
ENV PDM_CHECK_UPDATE=false
# copy files
COPY pyproject.toml pdm.lock README.md /problolm/
COPY src/ /problolm/src

# install dependencies and project into the local packages directory
WORKDIR /problolm
RUN pdm install --check -G:all --editable

# run stage
FROM python:$PYTHON_BASE

# retrieve packages from build stage
COPY --from=builder /problolm/.venv/ /problolm/.venv
ENV PATH="/problolm/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "problolm"]
