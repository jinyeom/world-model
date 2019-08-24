ARG CUDA_VERSION=10.1
ARG CUDNN_VERSION=7
ARG UBUNTU_VERSION=18.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl vim ca-certificates libjpeg-dev libpng-dev python-opengl
RUN rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Install required packages.
RUN conda install -y python=3.7 pip numpy scipy
RUN pip install matplotlib tqdm pillow seaborn cma

# Install the latest version of PyTorch.
RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN conda clean -ya

# Install the latest version of OpenAI Gym with Box2D.
RUN pip install gym[box2d]

# Configure working directory
WORKDIR /workspace
RUN chmod -R a+w /workspace

ENTRYPOINT ["python", "main.py"]