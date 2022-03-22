# onnx runtime: https://github.com/Microsoft/onnxruntime (brew install libomp)
# nice venv from requirements refresher: https://stackoverflow.com/questions/43256369/how-to-rename-a-virtualenv-in-python

import os
import onnx
import warnings
import numpy as np
import random
import onnxruntime as rt
from PIL import Image

def load_model(path:str) -> rt.InferenceSession:
    # generate an onnx runtime inference session based on the input filepath

    if os.path.exists(path):
        sess = rt.InferenceSession(path)
    else:
        assert False
    return sess


def run_inference(sess:rt.InferenceSession, TRUNCATION:float=0.7, seed=None) -> np.array:
    # return raw array from Generator

    # really badly set numpy seed from string if provided
    # note that `hash()` would not work here, as it is itself initialized from a random state
    # it would be possible to force hash() to be set from a static type, but would require an additional python file
    #   to set the environment seed... So equally awful
    random.seed(seed) # this may be a string
    np.random.seed(random.randint(0, 2**32-1)) # use python random state to generate numpy random state repeatably, lol
    
    # randomized generator inputs
    latents = np.random.randn(1,512).astype(np.float32)
    truncation = np.array([TRUNCATION]).astype(np.float32)

    input_name_a = sess.get_inputs()[0].name
    input_name_b = sess.get_inputs()[1].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name_a: latents, input_name_b:truncation})[0]

    return pred


def post_process_preds(pred:np.array) -> np.array:
    # format Generator output to actual image

    # strip batch dimension, move channels to back
    arr = np.moveaxis(np.squeeze(pred), 0, -1)
    # rescale 0-255, format as uint8
    arr_rescale = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

    return arr_rescale


def main():

    ONNX_MODEL = "./twdne3.onnx"
    TRUNCATION = 1.5

    # get session
    sess = load_model(ONNX_MODEL)

    # run inference
    pred = run_inference(sess, TRUNCATION)

    # post process
    arr = post_process_preds(pred)

    # save waifu
    im = Image.fromarray(arr)
    im.save("waifu.jpeg")
    print('Saved waifu!')


if __name__ == '__main__':
    main()
