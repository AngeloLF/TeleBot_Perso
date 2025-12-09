from telegram import Bot, InputFile
import asyncio
import coloralf as c
import sys, os
path_home = os.path.expanduser("~/")




async def main(text=None, image=None):

    TOKEN_file = f"{path_home}ccalf.txt"
    ID_file = f"{path_home}alfid.txt"

    with open(f"{TOKEN_file}", "r") as f : TOKEN = f.read()
    with open(f"{ID_file}", "r") as f : CHAT_ID = f.read()

    try:

        bot = Bot(TOKEN)
        bot_name_class = await bot.getMyName()
        bot_name = bot_name_class.name

        if text is not None:

            texts = text if isinstance(text, (list)) else [text]

            for t in texts:
                await bot.send_message(chat_id=CHAT_ID, text=f"{t}")
                print(f"{c.ti}<{bot_name}> send : {t}{c.d}")


        if image is not None:

            images = image if isinstance(image, (list)) else [image]

            for i in images:
                with open(i, 'rb') as photo_file:
                    await bot.send_photo(chat_id=CHAT_ID, photo=InputFile(photo_file))
                    print(f"{c.ti}<{bot_name}> send image : {i}{c.d}")

    except Exception as e:

        print(f"{c.r}WARNING [in telebot.py] : {e}{c.d}")


def send_message(text, folder_):

    asyncio.run(main(text=text))

def send_image(text=None, image=None):

    asyncio.run(main(text=text, image=image))




if __name__ == '__main__':
    

    if "test" in sys.argv[1:]:

        print(f"Begin test ...")

        if "message" in sys.argv[1:] or "m" in sys.argv[1:]:

            send_message("Test solo")
            send_message(["Test combiner...", 12.3, "... c'est bon"])

        if "image" in sys.argv[1:] or "s" in sys.argv[1:]:

            send_image(text="Voici chou : ", image=["./chou128.png"])

        print(f"Test end.")
    
    else:

        text = None
        TOKEN_file = None
        ID_file = None

        for argv in sys.argv[1:]:

            if argv[:4] == "msg=" : text = argv[4:]
            if argv[:6] == "token=" : TOKEN_file = argv[6:]
            if argv[:3] == "id=" : ID_file = argv[3:]

        with open(f"{TOKEN_file}", "r") as f : TOKEN = f.read()
        with open(f"{ID_file}", "r") as f : CHAT_ID = f.read()

        send_message(text)
