FROM ubuntu:22.04

# Setup conda environment
COPY --from=continuumio/miniconda3:23.5.2-0 /opt/conda /opt/conda

ENV PATH=/opt/conda/bin:$PATH

RUN apt update --allow-unauthenticated

RUN DEBIAN_FRONTEND="noninteractive" apt install -y git lz4 lsb-release wget software-properties-common \
    gnupg build-essential texlive dvipng texlive-latex-extra cm-super texlive-fonts-recommended \
    graphviz libstdc++-12-dev lcov

# Install racket
RUN wget https://plt.cs.northwestern.edu/racket-mirror/8.7/racket-8.7-x86_64-linux-cs.sh && \
    bash racket-8.7-x86_64-linux-cs.sh --unix-style --dest /usr --create-dir && \
    rm racket-8.7-x86_64-linux-cs.sh

RUN wget https://apt.llvm.org/llvm.sh && bash llvm.sh 14 && apt update --allow-unauthenticated

RUN DEBIAN_FRONTEND="noninteractive" apt install -y clang-14 libclang-14-dev llvm-14-dev libclang-rt-14-dev

RUN conda init bash

RUN conda create -n cov python=3.8 -y && conda create -n std python=3.8 -y

COPY ./ /artifact

SHELL ["conda", "run", "-n", "std", "/bin/bash", "-c"]

RUN bash /artifact/build/build_std.sh

SHELL ["conda", "run", "-n", "cov", "/bin/bash", "-c"]

RUN bash /artifact/build/build_cov.sh

ENV PYTHONPATH=/artifact/neuri:/artifact/

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
