import time
import logging
import asyncio
import glob
import os
from aiogram import Bot, Dispatcher, executor, types

from src.model import *

TOKEN = ""

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

@dp.message_handler(commands=["start"])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'{user_id} {user_full_name} {time.asctime()}')
    await message.reply(f"Hello, {user_full_name}! I'm able to make semantic segmentation of face areas in photo. Try it yourself!")

@dp.message_handler(content_types=["photo"])
async def photo_handler(message: types.Message):
    save_path = os.getcwd()+'/temp/'
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    image_name = save_path +'user_photo.png'
    predicted_image = save_path +'predict_photo.png'

    await message.photo[-1].download(destination_file=image_name)

    model = Model()
    model.predict(image_name, predicted_image)
    await bot.send_photo(message.chat.id, open(predicted_image, 'rb'))
    # remove products
    files = glob.glob(f'{save_path}*')
    for f in files:
        os.remove(f)

@dp.message_handler(content_types=["document", ])
async def photo_handler(message: types.Message):
    await message.reply(f"Send photo instead docs")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())