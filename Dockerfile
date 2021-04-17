FROM python:3.7-slim
# FROM ubuntu:20.04
WORKDIR /app

# install python
RUN apt update &&  apt install protobuf-compiler libprotoc-dev libomp-dev -y
#RUN apt install software-properties-common -y
#RUN add-apt-repository ppa:deadsnakes/ppa -y
#RUN apt install python3.7 -y
#RUN apt install python3-pip -y

# mount repo files
ADD . .

# install python and system requirements
RUN python3.7 -m pip install -r requirements.txt
