from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes
import os
from dotenv import load_dotenv
import asyncio
import logging

load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_IDS_STR = os.getenv('TELEGRAM_CHAT_IDS', '')

TARGET_CHAT_IDS = [int(id_str.strip()) for id_str in CHAT_IDS_STR.split(',') if id_str.strip()]

# Global dict to store acknowledged state per chat_id
acknowledged = {}

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Function to send buttons repeatedly until acknowledged for a specific chat_id
async def send_buttons_to_user(application, chat_id):
    global acknowledged
    # Initialize acknowledged state for this chat_id if not already set by a previous alert
    if chat_id not in acknowledged:
        acknowledged[chat_id] = False

    keyboard = [
        [
            InlineKeyboardButton("Stop Alerts 🛑", callback_data=f"ack_{chat_id}"), 
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Set to False for this specific alert cycle for this user
    acknowledged[chat_id] = False
    logger.info(f"Starting alert cycle for chat_id: {chat_id}. Acknowledged: {acknowledged.get(chat_id)}")


    while not acknowledged.get(chat_id, False): # Check specific chat_id
        try:
            await application.bot.send_message(chat_id=chat_id, text="Fall detected! ⚠️", reply_markup=reply_markup)
            logger.info(f"Alert sent to {chat_id}. Waiting for acknowledgment or next cycle.")
        except Exception as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
            # Optional: handle specific exceptions, e.g., if user blocked the bot
            acknowledged[chat_id] = True # Stop trying if there's a persistent error like bot blocked
            break
        await asyncio.sleep(10)  # wait before sending again
    
    if acknowledged.get(chat_id, False):
        logger.info(f"Alert cycle for chat_id: {chat_id} was acknowledged and stopped.")

# CallbackQuery handler
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global acknowledged
    query = update.callback_query
    user_chat_id = query.message.chat.id

    await query.answer()

    if query.data.startswith("ack_"):
        acknowledged[user_chat_id] = True
        await query.edit_message_text(text=f"Alerts are paused for you!")
        logger.info(f"Alerts acknowledged and paused by chat_id: {user_chat_id}")
    
# Run on bot startup
async def on_startup(application: Application):
    if not TARGET_CHAT_IDS:
        logger.warning("No TARGET_CHAT_IDS configured. Alerts will not be sent.")
        return
    
    for chat_id in TARGET_CHAT_IDS:
        logger.info(f"Initializing alert task for chat_id: {chat_id}")
        acknowledged[chat_id] = False
        asyncio.create_task(send_buttons_to_user(application, chat_id))

# Main function
def main():
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables.")
        return
    if not TARGET_CHAT_IDS:
        logger.error("TELEGRAM_CHAT_IDS not found or empty in environment variables. Please set it to a comma-separated list of chat IDs.")
        return
        
    application = Application.builder().token(TOKEN).post_init(on_startup).build()

    application.add_handler(CallbackQueryHandler(button)) # Handles any callback

    logger.info("Telegram Bot started. Polling for updates...")
    application.run_polling()

if __name__ == "__main__":
    main()