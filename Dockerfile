FROM python:3.7-slim
# FROM ubuntu:20.04
WORKDIR /app

# install python
RUN apt update &&  apt install protobuf-compiler libprotoc-dev libomp-dev -y

# mount repo files
ADD *.py .
ADD aws aws
ADD coolvetica coolvetica
ADD josephsophia josephsophia
ADD twdne3.onnx .
add requirements.txt .

# install python and system requirements
RUN python3.7 -m pip install -r requirements.txt
