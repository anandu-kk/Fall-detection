from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
USER_CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID'))

# Global dict to store acknowledged state per chat
acknowledged = {}

# Function to send buttons repeatedly until acknowledged
async def send_buttons(application, chat_id):
    global acknowledged
    acknowledged[chat_id] = False

    keyboard = [
        [
            InlineKeyboardButton("Stop Alerts 🛑", callback_data="1"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    while not acknowledged.get(chat_id, False):
        await application.bot.send_message(chat_id=chat_id, text="fall detected! ⚠️", reply_markup=reply_markup)
        await asyncio.sleep(5)  # wait before sending again


# CallbackQuery handler
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    global acknowledged
    query = update.callback_query
    chat_id = query.message.chat.id
    await query.answer()

    if query.data == "1":
        acknowledged[chat_id] = True
        await query.edit_message_text(text="Alerts are paused !")
    
# Run on bot startup
async def on_startup(application: Application):
    asyncio.create_task(send_buttons(application, USER_CHAT_ID))

# Main function
def main():
    application = Application.builder().token(TOKEN).post_init(on_startup).build()

    application.add_handler(CallbackQueryHandler(button))

    application.run_polling()

if __name__ == "__main__":
    main()
