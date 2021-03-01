FROM ubuntu:16.04

# Installing python 2.7 and 3.5 (requirement by snpe setup)
RUN apt-get update && apt-get install -y \
    python \
    python3 \
    python3-pip \
    unzip \ 
    sudo \
    wget \
    apt-utils \
    zip

# Specified in https://developer.qualcomm.com/docs/snpe/setup.html
RUN apt-get update && \
    update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.5 2 && \
    update-alternatives --config python

# Check pillow compatibility for 3.5: https://pillow.readthedocs.io/en/stable/installation.html
RUN pip3 install \
    numpy==1.16.5 \
    sphinx==2.2.1 \
    scipy==1.3.1 \
    matplotlib==3.0.3 \
    protobuf==3.6.0 \
    pillow==7.2.0 \
    scikit-image==0.15.0 \
    pyyaml==5.1 \
    mako==1.1.4 \
    onnx==1.3.0

COPY snpe-1.47.0.zip /tmp/

RUN cd /tmp && unzip snpe-1.47.0.zip

RUN apt-get update && apt-get install -y \
    python3-dev \
    python-dev \
    adb \
    vim

RUN cd /tmp && \
    /bin/bash -c "source snpe-1.47.0.2501/bin/dependencies.sh" && \
    /bin/bash -c "source snpe-1.47.0.2501/bin/check_python_depends.sh"

ENV SNPE_ROOT /tmp/snpe-1.47.0.2501
ENV ONNX_HOME /usr/local/lib/python3.5/dist-packages/onnx

RUN cd $SNPE_ROOT && \
    /bin/bash -c "source bin/envsetup.sh -o $ONNX_HOME"

# get directory of the bash script
ENV SOURCEDIR /tmp/snpe-1.47.0.2501/bin
ENV SNPE_ROOT /tmp/snpe-1.47.0.2501
ENV PATH $SNPE_ROOT/bin/x86_64-linux-clang:$PATH
# setup LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
# setup PYTHONPATH
ENV PYTHONPATH $SNPE_ROOT/lib/python:$PYTHONPATH
ENV PYTHONPATH $SNPE_ROOT/models/lenet/scripts:$PYTHONPATH
ENV PYTHONPATH $SNPE_ROOT/models/alexnet/scripts:$PYTHONPATH

#setup SNPE_UDO_ROOT
ENV SNPE_UDO_ROOT $SNPE_ROOT/share/SnpeUdo/

CMD ["/bin/bash"]