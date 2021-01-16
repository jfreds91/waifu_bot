#!/home/ubuntu/waifu_bot/waifu_bot/bin/python
# async routines: https://dev.to/code_enzyme/introduction-to-using-async-await-in-python-2i0n
# discord client event reference: https://discordpy.readthedocs.io/en/latest/api.html

import discord
import configparser
import generator
from PIL import Image
from discord.ext import commands
import numpy as np
import tempfile
from datetime import datetime
from aws import secret_reader

config = configparser.ConfigParser()
config.read('./credentials.cfg')
token = config['discord']['token']

# prepare client and bot object
bot = commands.Bot(command_prefix='$')
# prepare model
sess = generator.load_model("./twdne3.onnx")
print("Successfully loaded model")


@bot.event
async def on_ready():
    # print(f'We have logged in as {client.user}')
    print(f'We have logged in as {bot.user}')


# @bot.event
# async def on_message(message):

#     if message.author == bot.user:
#         # self message, do nothing
#         return
    
#     if message.content.lower().startswith('claim waifu'):
#         await message.channel.send(f'Recieved call for waifu from {message.author}...')
        
#         # generate waifu
#         # run inference
#         pred = generator.run_inference(sess, TRUNCATION=.75)
#         # post process
#         arr = generator.post_process_preds(pred)
#         # save waifu
#         im = Image.fromarray(arr)
#         im.save(filename)

#         await message.channel.send(file=discord.File(filename))


# bot arguments
@bot.command()
async def test(ctx, arg=None):
    # simply repeat the argument to the context (channel)
    if arg:
        await ctx.send(arg)
    else:
        await ctx.send('No arg supplied')   

@bot.command()
async def claim_waifu_truncation(ctx, arg='0.75'):
    # claim waifu at desired truncation
    
    await ctx.send(f'Received call for waifu from {ctx.message.author}...')

    if type(arg) == float:
        trunc = arg
    else:
        # user specified input
        try:
            trunc = float(arg)
        except ValueError as e:
            await ctx.send(f'Sowwy I can\'t convert {arg} into a float')
            return
        except Exception as e:
            await ctx.send(f'@jfreds, {ctx.message.author} is trying to bweak me! :( \n{e}')
            return
        if not np.isfinite(trunc):
            await ctx.send(f'Hey {ctx.message.author}, fucking stop nerd')
            return

    # generate waifu
    # run inference
    pred = generator.run_inference(sess, TRUNCATION=trunc)
    # post process
    arr = generator.post_process_preds(pred)
    # save waifu
    im = Image.fromarray(arr)
    temp = tempfile.NamedTemporaryFile(suffix=".jpeg")
    im.save(temp)
    print(f"Saved {temp.name}")

    await ctx.send(
        content=f'intensity = {trunc}',
        file=discord.File(temp.name)
        )
    temp.close()


@bot.command()
async def claim_waifu(ctx, arg=None):
    # claim waifu with desired name
    
    await ctx.send(f'Received call for waifu from {ctx.message.author}...')

    # generate waifu
    # run inference
    print(f"Running inference for {ctx.message.author} at {datetime.now()}")
    pred = generator.run_inference(sess, seed=arg)
    # post process
    print("Post-processing")
    arr = generator.post_process_preds(pred)
    # save waifu in named temporary file. Name is only important to let PIL know what format to use
    im = Image.fromarray(arr)
    temp = tempfile.NamedTemporaryFile(suffix=".jpeg")
    im.save(temp)
    print(f"Saved {temp.name}")

    await ctx.send(
        content=f'Meet {arg}-chan, isn\'t (s)he beautiful?' if arg else None,
        file=discord.File(temp.name)
        )

    temp.close()


bot.run(token)
