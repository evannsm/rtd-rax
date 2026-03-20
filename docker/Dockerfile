FROM osrf/ros:jazzy-desktop-full@sha256:b2f19f30c728c61b92cb3b8dd83928d2225e0dae04c1d2ed9b012f46b56ee16c

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/rtd-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /workspace/ws_RTD

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-tk \
    git \
    make \
    libcdd-dev \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /tmp/requirements.txt

# immrax uses a git submodule declared with an SSH URL; rewrite to HTTPS for
# non-interactive Docker builds, then install from the local checkout.
RUN git clone https://github.com/gtfactslab/immrax.git /tmp/immrax && \
    cd /tmp/immrax && \
    sed -i 's|git@github.com:|https://github.com/|g' .gitmodules && \
    git submodule sync --recursive && \
    git submodule update --init --recursive && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir /tmp/immrax && \
    rm -rf /tmp/immrax

COPY . /workspace/ws_RTD

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source ${VIRTUAL_ENV}/bin/activate" >> /root/.bashrc

CMD ["/bin/bash"]
