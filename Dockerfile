#FROM python:3.7i
FROM ubuntu:20.04
WORKDIR /app

# install python
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.7 -y
RUN apt install python3-pip -y

# mount repo files
ADD . .

# copy repo files over
# COPY aws aws
# COPY credentials.cfg credentials.cfg
# COPY discord_client.py discord_client.py
# COPY generator.py generator.py
# COPY requirements.txt requirements.txt
# COPY twdne3.onnx twdne3.onnx
# COPY waifu_bot waifu_bot

# install python and system requirements
RUN python3.7 -m pip install -r requirements.txt
RUN apt install protobuf-compiler libprotoc-dev libomp-dev -y
