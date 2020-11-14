# onnx runtime: https://github.com/Microsoft/onnxruntime (brew install libomp)
# nice venv from requirements refresher: https://stackoverflow.com/questions/43256369/how-to-rename-a-virtualenv-in-python

import os
import onnx
import warnings
import numpy as np
import onnxruntime as rt
from PIL import Image

def load_model(path:str) -> rt.InferenceSession:
    # generate an onnx runtime inference session based on the input filepath

    if os.path.exists(path):
        sess = rt.InferenceSession(path)
    else:
        assert False
    return sess


def run_inference(sess:rt.InferenceSession, TRUNCATION:float=0.7) -> np.array:
    # return raw array from Generator

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

    ONNX_MODEL = "/Users/jesse/Documents/waifu_bot/twdne3.onnx"
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