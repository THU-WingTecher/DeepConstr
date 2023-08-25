FROM ubuntu:20.04

RUN apt update --allow-unauthenticated

# Install racket
RUN wget https://plt.cs.northwestern.edu/racket-mirror/8.7/racket-8.7-x86_64-linux-cs.sh && \
    bash racket-8.7-x86_64-linux-cs.sh --unix-style --dest /usr --create-dir && \
    rm racket-8.7-x86_64-linux-cs.sh

RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.8-dev python3.8-venv python3-numpy \
    git lz4 lsb-release wget software-properties-common gnupg build-essential \
    texlive dvipng texlive-latex-extra cm-super texlive-fonts-recommended graphviz

RUN wget https://apt.llvm.org/llvm.sh && bash llvm.sh 14 && apt update --allow-unauthenticated

RUN DEBIAN_FRONTEND="noninteractive" apt install -y clang-14 libclang-14-dev llvm-14-dev

COPY ./ /artifact

RUN python3 -m venv /artifact/std && python3 -m venv /artifact/cov

RUN . /artifact/std/bin/activate && bash /artifact/build/build_std.sh

# Copy pre-built coverage instrumented wheels

RUN . /artifact/std/bin/activate && bash /artifact/build/build_cov.sh

ENV PYTHONPATH=/artifact/neuri
