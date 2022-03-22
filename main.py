import argparse
import generator
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
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
parser.add_argument('-w', dest='watermark', action='store_true',
	help='boolean for if waifubot will apply a watermark and increase saturation')

# handle arguments
args = vars(parser.parse_args())
if isinstance(args['seed'], list):
	args['seed'] = ' '.join(args['seed'])

# get generator specific args
gen_args = {
	'TRUNCATION':args['TRUNCATION'],
	'seed':args['seed']
}

### generate waifu ###
# get session
sess = generator.load_model('./twdne3.onnx')

# run inference
pred = generator.run_inference(sess, **{k: v for k, v in gen_args.items() if v is not None})

# post process
arr = generator.post_process_preds(pred)
im = Image.fromarray(arr)

if args['watermark']:
	# apply watermark
	im = ImageEnhance.Color(im).enhance(1.4)
	draw = ImageDraw.Draw(im)

	def shadow_text(draw, text, font_ttf, fontsize, x, y, offset=3):
		# adds text in both black and white to add a shadow
		# modifies draw object inplace
		font = ImageFont.truetype(font_ttf, fontsize)
		draw.text((x+offset, y+offset),text,(0,0,0),font=font)
		draw.text((x, y),text,(255,255,255),font=font)
	
	# add title
	name = args['seed']
	x_title = 20
	y_title = 450
	fontsize_title = 50
	font_title = os.path.join(os.path.dirname(os.path.abspath(__file__)),'josephsophia/josephsophia.ttf')

	author = '@OGWaifuBot'
	x_auth = 20
	y_auth = 480
	fontsize_auth = 20
	font_auth = os.path.join(os.path.dirname(os.path.abspath(__file__)),'coolvetica/coolvetica rg.ttf')

	if name:
	    shadow_text(draw, name, font_title, fontsize_title, x_title, y_title, 2)
	# shadow_text(draw, author, font_auth, fontsize_auth, x_auth, y_auth, 2)

# save waifu
filename = 'docker_waifu'
if gen_args['seed'] is not None:
	filename = gen_args['seed']
output_filepath = os.path.join(args['dir'],f'{filename}.jpeg')
im.save(output_filepath)
print(f'saved: {output_filepath}')
