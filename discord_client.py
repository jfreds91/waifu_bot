# async routines: https://dev.to/code_enzyme/introduction-to-using-async-await-in-python-2i0n
# discord client event reference: https://discordpy.readthedocs.io/en/latest/api.html

import discord
import configparser
import generator
from PIL import Image

config = configparser.ConfigParser()
config.read('./credentials.cfg')
token = config['discord']['token']

def client_startup():
    '''
    This function starts the discord client, loads the onnx model for inference,
        and defines client events
    '''
    global client
    global filename
    global sess

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
            pred = generator.run_inference(sess, TRUNCATION=.75)
            # post process
            arr = generator.post_process_preds(pred)
            # save waifu
            im = Image.fromarray(arr)
            im.save(filename)

            await message.channel.send(file=discord.File(filename))


if __name__ == '__main__':
    client_startup()
    client.run(token)