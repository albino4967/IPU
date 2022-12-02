FROM graphcore/pytorch:3.0.0-ubuntu-18.04
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y make cmake virtualenv git nano tree htop neofetch
RUN apt-get install -y libjpeg-turbo8-dev libffi-dev nasm libssl-dev rapidjson-dev libre2-dev libboost-all-dev
RUN git clone https://github.com/graphcore/examples.git mlcommons

WORKDIR /mlcommons/vision/cnns/pytorch
COPY poptorch-3.0.0+86949_163b7ce462_ubuntu_18_04-cp36-cp36m-linux_x86_64.whl .

RUN make install-turbojpeg
RUN virtualenv -p python3 --system-site-packages /venv
RUN source /venv/bin/activate && pip install -r requirements.txt
RUN pip install poptorch-3.0.0+86949_163b7ce462_ubuntu_18_04-cp36-cp36m-linux_x86_64.whl
