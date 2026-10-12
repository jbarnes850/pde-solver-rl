FROM python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579

LABEL org.opencontainers.image.title="RLVP source-derived solver sandbox" \
      org.opencontainers.image.description="Independent multi-PDE verifier sandbox; not an original RLVP artifact"

RUN pip install --no-cache-dir \
    numpy==2.5.2 \
    scipy==1.17.0

COPY src/rlvp/runner/execute_candidate.py /opt/rlvp/execute_candidate.py

RUN useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin solver \
    && chmod 0444 /opt/rlvp/execute_candidate.py

USER 65532:65532
WORKDIR /work
