import argparse
import generator
import tempfile
from PIL import Image

parser = argparse.ArgumentParser(description='waifubot CLI')
parser.add_argument('-s', type=str, dest='seed',
                    help='optional seed string')
parser.add_argument('-t', type=float, dest='trunc',
                    help='truncation value to use. Defaults to None to use package default')

args = vars(parser.parse_args())

# get session
sess = generator.load_model('./twdne3.onnx')

# generate waifu
# run inference
if args['trunc']:
	pred = generator.run_inference(sess)
else:	
	pred = generator.run_inference(sess, TRUNCATION=args['trunc'])
# post process
arr = generator.post_process_preds(pred)
# save waifu
im = Image.fromarray(arr)
temp = tempfile.NamedTemporaryFile(suffix=".jpeg")
im.save(temp)
print(f"Saved {temp.name}")
