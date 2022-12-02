FROM graphcore/pytorch:3.0.0-ubuntu-18.04
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y make cmake virtualenv git nano tree htop neofetch sudo vim
RUN apt-get install -y libsndfile1 sox libboost-dev

RUN git clone https://github.com/graphcore/examples.git

WORKDIR /examples/speech/transformer_transducer/popart/
COPY horovod-1.3.14+33691.f5968f9ec9-cp36-cp36m-linux_x86_64.whl .

RUN virtualenv -p python3 --system-site-packages /rnnt_venv
RUN source /rnnt_venv/bin/activate

RUN make all
RUN pip install -r training/requirements.txt
RUN pip install horovod-1.3.14+33691.f5968f9ec9-cp36-cp36m-linux_x86_64.whl