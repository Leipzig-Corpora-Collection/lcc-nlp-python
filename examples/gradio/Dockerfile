# FROM python:3.12.1-slim-bookworm
FROM python:3.12.1-alpine3.19

ENV PIP_ROOT_USER_ACTION=ignore
# home for "nobody" user (e.g., caches, configs)
ENV HOME=/workspace
# gradio stuff
ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}
ENV GRADIO_NUM_PORTS=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_ANALYTICS_ENABLED=False
# whether file paths to tool config files can be set
ENV READONLY_PATHS=True

EXPOSE ${GRADIO_SERVER_PORT}

WORKDIR /workspace

# standard pip update
RUN python3 -m pip install -U pip setuptools wheel

# resources
COPY resources /workspace/resources

# install LCC-NLP tools
COPY setup.cfg setup.py pyproject.toml MANIFEST.in LICENSE /workspace/
COPY src /workspace/src
RUN python3 -m pip install .

# add gradio stuff
COPY examples/gradio /workspace/gradio
RUN python3 -m pip install -r /workspace/gradio/requirements.txt

# downgrade user
RUN chmod -R o+rw /workspace
USER nobody

CMD ["python3", "/workspace/gradio/lcc_demo.py"]
