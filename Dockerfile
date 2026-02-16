FROM ubuntu:latest
WORKDIR /app/

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip3 && \
    pip3 install pdm

RUN pdm config venv.in_project true

COPY . /app/

RUN pdm install -G:all

# Set the default command to run the Python script, but allow overriding
ENTRYPOINT ["pdm", "run", "python", "-m", "problolm"]
