import discord
import asyncio
import coloralf as c
import sys, os
path_home = os.path.expanduser("~/")




async def send_message_async(client, CHAT_ID, text=None):

    try:

        user = await client.fetch_user(CHAT_ID)

        if text is not None:

            texts = text if isinstance(text, (list)) else [text]

            if user:
                for t in texts:
                    await user.send(t)
                    print(f"{c.ti}<{client.user.name}#{client.user.discriminator}> send : {t}{c.d}")
            else:
                print(f"WARNING: user {CHAT_ID} non trouvé.")

    except Exception as e:

        print(f"{c.r}WARNING [in discobot.py] : {e}{c.d}")

    finally:
        print(f"Shut-down {client.user.name}#{client.user.discriminator}...")
        await client.close()
        


def send_message(text):

    TOKEN_file = f"{path_home}bot_dc.txt"
    ID_file = f"{path_home}alfid_dc.txt"

    with open(f"{TOKEN_file}", "r") as f : TOKEN = f.read()
    with open(f"{ID_file}", "r") as f : CHAT_ID = f.read()

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        # print(f'Connecté en tant que {client.user.name}#{client.user.discriminator}')
        await send_message_async(client, CHAT_ID, text)

    client.run(TOKEN)




if __name__ == '__main__':
    

    if "test" in sys.argv[1:]:

        print(f"Begin test ...")

        if "message" in sys.argv[1:] or "m" in sys.argv[1:]:

            send_message("Test solo")
            send_message(["Test combiner...", 12.3, "... c'est bon"])

        print(f"Test end.")
    
    else:

        text = None

        for argv in sys.argv[1:]:

            if argv[:4] == "msg=" : text = argv[4:]

        send_message(text)
