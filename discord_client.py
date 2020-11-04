# async routines: https://dev.to/code_enzyme/introduction-to-using-async-await-in-python-2i0n

import discord
import configparser
import generator
from PIL import Image

config = configparser.ConfigParser()
config.read('./credentials.cfg')
token = config['discord']['token']

# prepare client
client = discord.Client()
filename = './discord_waifu.jpeg'
# prepare model
sess = generator.load_model("/Users/jesse/Documents/waifu_bot/twdne3.onnx")

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        # self message, do nothing
        return
    
    if message.content.lower().startswith('claim waifu'):
        await message.channel.send(f'Recieved call for waifu from {message.author}...')
        
        # generate waifu
        # run inference
        pred = generator.run_inference(sess, TRUNCATION=.8)
        # post process
        arr = generator.post_process_preds(pred)
        # save waifu
        im = Image.fromarray(arr)
        im.save(filename)

        await message.channel.send(file=discord.File(filename))



client.run(token)