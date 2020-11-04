# Waifu_Bot
Jesse Fredrickson
11/3/2020

![waifu_bot in action](disc_ex.png)

## Purpose
This repo contains the code to operate simple a discord bot which, when prompted with a "claim waifu" message in chat, will dynamically generate and post a unique anime girlfriend. Who wouldn't want that

## Usage
It is assumed the user already has python3. A basic understanding of python is useful but not necessary.

**Python Environment**
- Clone this repository `git clone https://github.com/jfreds91/waifu_bot`
- Navigate to cloned repo and create a new virtual environment `python3 -m venv env`
- Activate environment `source env/bin/activate`
- Update environment from requirements doc `pip install -r requirements.txt`

**System Installs**
- At this point you already have the onnx runtime package(s) installed, but you may have to install additional dependencies, see below. On MacOS, I had to install OpenMP `brew install libomp`
- Follow the install instructions at https://github.com/Microsoft/onnxruntime

**GAN Model download**
- Download the TWDNE3.onnx model from https://hivemind-repo.s3-us-west-2.amazonaws.com/twdne3/twdne3.onnx and move it to your project repo
- If that's no longer active, look for an update on https://www.gwern.net/Faces#

## Motivation
The application is fun, but I wrote this project to explore a few interesting technical areas.
1. ONNX. [ONNX](https://onnx.ai) is a common model interoperability platform introduced by Facebook and Microsoft in 2017, and allows for users to run framework-agnostic models by converting to its own common format. This repo uses an onnx model and onnx runtime to perform inference and generate images.
2. Discord API. I have a small but active discord which has helped me get through quarantine (COVID19), and improving it (for a loose definition of improve) is intriguing
3. GANs. Though I did not train this model myself, learning how to (ab)use the generator model was a learning experience, and Gwern's blog was a fascinating read.

Thanks for reading, enjoy!
