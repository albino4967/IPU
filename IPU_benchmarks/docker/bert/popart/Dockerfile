FROM graphcore/pytorch:3.0.0-ubuntu-18.04
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y make cmake virtualenv wget git nano tree htop neofetch
RUN git clone https://github.com/graphcore/examples.git mlcommons

WORKDIR /mlcommons/nlp/bert/popart
COPY tensorflow-1.15.5+gc3.0.0+236839+f53da99dba1+amd_znver1-cp36-cp36m-linux_x86_64.whl .

RUN make
RUN virtualenv -p python3 --system-site-packages /venv
RUN source /venv/bin/activate && pip install -r requirements.txt
RUN pip install tensorflow-1.15.5+gc3.0.0+236839+f53da99dba1+amd_znver1-cp36-cp36m-linux_x86_64.whl

COPY download_tf_checkpoints.sh .
RUN chmod +x download_tf_checkpoints.sh
RUN source /venv/bin/activate && ./download_tf_checkpoints.sh