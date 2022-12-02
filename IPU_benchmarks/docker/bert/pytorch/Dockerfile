FROM graphcore/pytorch:3.0.0-ubuntu-18.04
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y make cmake virtualenv wget git
RUN apt-get install -y nano tree htop neofetch sudo vim
RUN apt-get install -y libboost-all-dev build-essential libre2-dev libssl-dev python3-dev rapidjson-dev

RUN git clone https://github.com/graphcore/examples.git

WORKDIR /examples/nlp/bert/pytorch/
COPY poptorch-3.0.0+86949_163b7ce462_ubuntu_18_04-cp36-cp36m-linux_x86_64.whl .

RUN virtualenv -p python3 --system-site-packages /venv
RUN source /venv/bin/activate

RUN	pip install -r requirements.txt
RUN pip install poptorch-3.0.0+86949_163b7ce462_ubuntu_18_04-cp36-cp36m-linux_x86_64.whl
RUN HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch]