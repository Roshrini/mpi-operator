FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.4.0-gpu-py36-cu90-ubuntu16.04 AS build

# Official mxnet pip package built with gcc4
RUN pip install mxnet-cu90==1.4.0.post0

# Install cuDNN, NCCL
RUN apt-get update && apt-get install -y  --allow-downgrades --allow-change-held-packages --no-install-recommends \
        cuda-cublas-dev-9-0 \
        cuda-cufft-dev-9-0 \
        cuda-curand-dev-9-0 \
        cuda-cusolver-dev-9-0 \
        cuda-cusparse-dev-9-0 \
        libcudnn7=7.3.1.20-1+cuda9.0 \
        libnccl2=2.3.5-2+cuda9.0 \
        libnccl-dev=2.3.5-2+cuda9.0

# Install gcc4.9 to install horovod with gcc4.9 instead of default gcc5
RUN apt-get install -y --no-install-recommends \
        openssh-client gcc-4.9 g++-4.9 gcc-4.9-base software-properties-common

# --------------------------------------------------------------------
#### Needed for horovod ####
# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Pin GCC to 4.9 to compile correctly against MXNet
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 100

RUN update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 100

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 100

RUN update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 100

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod==0.16.1 && \
    ldconfig

# Remove GCC pinning
RUN update-alternatives --remove gcc /usr/bin/gcc-4.9

RUN update-alternatives --remove x86_64-linux-gnu-gcc /usr/bin/gcc-4.9

RUN update-alternatives --remove g++ /usr/bin/g++-4.9

RUN update-alternatives --remove x86_64-linux-gnu-g++ /usr/bin/g++-4.9

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# --------------------------------------------------------------------

# Other packages needed for running MXNet-horovod distributed training examples
RUN pip install gluoncv

# Clone Horovod source to run examples
RUN git clone -b v0.16.1 https://github.com/horovod/horovod/

WORKDIR "/"
CMD ["bin/bash"]
