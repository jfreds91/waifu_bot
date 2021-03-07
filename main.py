import argparse
import generator
from PIL import Image
import os

# the purpose of this file is to handle arguments passed via docker run
#	to generate a named waifu inside a container and save it back to 
#	the host.

# the command should be:
# docker run -v "$(pwd)":/mnt/here jfreds/waifu_docker_test:v0 python3.7 /mnt/here/main.py -dir /mnt/here/

parser = argparse.ArgumentParser(description='waifubot CLI')
parser.add_argument('-s', type=str, dest='seed', nargs='+',
	help='optional seed string')
parser.add_argument('-t', type=float, dest='TRUNCATION',
	help='truncation value to use. Defaults to None to use package default')
parser.add_argument('-dir', type=str, dest='dir', default='.',
	help='Where to store output image. Sould be a mounted directory')

args = vars(parser.parse_args())
if isinstance(args['seed'], list):
	args['seed'] = ' '.join(args['seed'])

# get session
sess = generator.load_model('./twdne3.onnx')

# get generator specific args
gen_args = {'TRUNCATION':args['TRUNCATION'], 'seed':args['seed']}

### generate waifu ###

# run inference
pred = generator.run_inference(sess, **{k: v for k, v in gen_args.items() if v is not None})

# post process
arr = generator.post_process_preds(pred)

# save waifu
im = Image.fromarray(arr)
filename = 'docker_waifu'
if gen_args['seed'] is not None:
	filename = gen_args['seed']
output_filepath = os.path.join(args['dir'],f'{filename}.jpeg')
im.save(output_filepath)
print(f'saved: {output_filepath}')
