# async routines: https://dev.to/code_enzyme/introduction-to-using-async-await-in-python-2i0n
# discord client event reference: https://discordpy.readthedocs.io/en/latest/api.html

import discord
import configparser
import generator
from PIL import Image
from discord.ext import commands
import numpy as np

config = configparser.ConfigParser()
config.read('./credentials.cfg')
token = config['discord']['token']

# prepare client and bot object
#client = discord.Client()
filename = './discord_waifu.jpeg'
bot = commands.Bot(command_prefix='$')
# prepare model
sess = generator.load_model("/Users/jesse/Documents/waifu_bot/twdne3.onnx")


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
async def claim_waifu(ctx, arg='0.75'):
    # claim waifu at desired truncation

    await ctx.send(f'Recieved call for waifu from {ctx.message.author}...')

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
            await ctx.send(f'I\'m sowwy {ctx.message.author}, I can\'t let you do that')
            return

    # generate waifu
    # run inference
    pred = generator.run_inference(sess, TRUNCATION=trunc)
    # post process
    arr = generator.post_process_preds(pred)
    # save waifu
    im = Image.fromarray(arr)
    im.save(filename)

    await ctx.send(file=discord.File(filename))


bot.run(token)