FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8"

ARG project_name=sovitssvc
ARG uid=1001
ARG gid=1001
ARG username=mluser
ARG APPLICATION_DIRECTORY=/home/${username}/${project_name}

RUN echo "uid ${uid}"
RUN echo "gid ${gid}"
RUN echo "username ${username}"
RUN groupadd -r -f -g ${gid} ${username} && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash ${username}

RUN apt-get update -y && apt-get install -y build-essential vim \
    wget curl git zip gcc make cmake openssl \
    libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev python3-tk tk-dev python-tk \
    libfreetype6-dev libffi-dev liblzma-dev libsndfile1 ffmpeg -y

USER ${username}
WORKDIR ${APPLICATION_DIRECTORY}

# python関連
RUN git clone https://github.com/yyuu/pyenv.git /home/${username}/.pyenv
ENV HOME /home/${username}
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN ls $PYENV_ROOT/bin
RUN pyenv --version

RUN pyenv install 3.10.11
RUN pyenv global 3.10.11

RUN python --version
RUN pyenv rehash
RUN pip install --upgrade pip setuptools requests
RUN pip install poetry