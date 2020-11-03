# test onnx image gen

# pkl version of twdne3: https://www.gwern.net/TWDNE#downloads
# rsync used: rsync --recursive --times --verbose rsync://78.46.86.149:873/twdne/2019-02-26-stylegan-faces-network-02048-016041.pkl ./
# simple onnx-tf example: https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb
# pkl version of twdne example: https://github.com/Antiky/StyleGAN-Anime/blob/master/pretrained_example.py
# dnnlib needed for pkl: pip install https://github.com/podgorskiy/dnnlib/releases/download/0.0.1/dnnlib-0.0.1-py3-none-any.whl
# onnx runtime: https://github.com/Microsoft/onnxruntime (brew install libomp)


import onnx
import warnings
# from onnx_tf.backend import prepare
import numpy as np
# import pickle
# import dnnlib
# import dnnlib.tflib as tflib
# import tensorflow as tf
import onnxruntime as rt
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':

    USE_ONNX = True
    USE_PKL = False
    TRUNCATION = .7

    # randomized generator inputs
    latents = np.random.randn(1,512).astype(np.float32)
    truncation = np.array([TRUNCATION]).astype(np.float32)

    # ONNX
    if USE_ONNX:

        sess = rt.InferenceSession("/Users/jesse/Documents/waifu_bot/twdne3.onnx")
        input_name_a = sess.get_inputs()[0].name
        input_name_b = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred = sess.run([label_name], {input_name_a: latents, input_name_b:truncation})[0]

        # strip dimensions
        arr = np.moveaxis(np.squeeze(pred), 0, -1)
        # rescale
        arr_rescale = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        # save waifu
        im = Image.fromarray(arr_rescale)
        im.save("waifu.jpeg")
        print('Saved waifu!')
    
    # pkl
    if USE_PKL:
        tflib.init_tf()
        with open('/Users/jesse/Documents/waifu_bot/2019-02-26-stylegan-faces-network-02048-016041.pkl', 'rb') as f:
            model = pickle.load(f)
        print('x')
