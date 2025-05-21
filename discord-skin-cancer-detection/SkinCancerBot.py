import discord
from discord.ext import commands
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
import numpy as np
from urllib.request import urlopen
import requests
from PIL import Image
from io import BytesIO

LABELS = [
    'Actinic Keratoses and intraephithelial carcinoma / Bowen\'s Disease :reminder_ribbon: ',
    'Basal Cell Carcinoma :reminder_ribbon: ',
    'Beningn keratosis-like lesions',
    'Dermatofibroma',
    'Melanoma :reminder_ribbon: ',
    'Melanocytic Nevi',
    'Vascular Lesions'
]
model = load_model("SCancer_invbotres_2435K.h5", compile=False)

def get_pred(stream):
    im = Image.open(stream)
    im = im.resize([224,224])
    im = np.array(im.convert("RGB"))
    im = tf.expand_dims(im, 0)
    return model.predict(im, verbose=0)[0]

client = commands.Bot(command_prefix='.', intents=discord.Intents.all())
client.intents.message_content = True

@client.event
async def on_ready():
    print("Bot is up")

@client.command()
async def cancer(ctx):
    try:
        url = ctx.message.attachments[0].url
    except IndexError:
        await ctx.channel.send("Please provide an image")
    else:
        if url[0:26] == "https://cdn.discordapp.com":
            file = ctx.message.attachments[0]
            fp = BytesIO()
            await file.save(fp)
            fp.seek(0)
            pred = get_pred(fp)
            top1 = np.argmax(pred)
            top2 = int(np.argpartition(pred, -2, axis=0)[-2])

            if pred[top1] < 0.5:
                await ctx.channel.send(f"Im not sure")
            else:
                await ctx.channel.send(f"Here are the Top 2 results :")
                await ctx.channel.send(f"{LABELS[top1]} : {round(pred[top1]*100, 2)}%")
                await ctx.channel.send(f"{LABELS[top2]} : {round(pred[top2]*100, 2)}%")

client.run('')
