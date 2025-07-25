# VetBot: A Multimodal AI Veterinarian Assistant for Telegram
# This script uses the python-telegram-bot and google-generativeai libraries.

import os
import logging
import asyncio
import google.generativeai as genai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction
from telegram.error import BadRequest

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up basic logging to monitor the bot's activity and errors.
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Gemini AI Setup ---
# This is the core instruction that defines the bot's persona and rules.
SYSTEM_INSTRUCTION = """
You are VetBot, a specialized AI assistant designed to help pet owners.
Your goal is to provide preliminary information and general advice about pet health, behavior, and care based on the text, images, audio, or video provided by the user.

When responding:
1.  Adopt a caring, empathetic, and professional tone.
2.  Analyze the provided media carefully. If it's an image, describe what you see. If it's audio, describe the sound. If it's video, describe the actions.
3.  Provide potential explanations or general advice related to the user's query. Do not give a definitive diagnosis.
4.  Suggest general care tips or next steps the owner might consider.
5.  **Crucial Disclaimer:** ALWAYS end every single response with the following disclaimer, formatted exactly like this on a new line:

---
**⚠️ Disclaimer:** I am an AI assistant and not a licensed veterinarian. My advice is for informational purposes only and is not a substitute for a professional veterinary diagnosis or treatment. Please consult a qualified vet for any health concerns with your pet.
"""

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    # Exit if Gemini is not configured, as the bot cannot function.
    exit()

# Dictionary to hold separate conversation histories for each user.
# Key: user_id, Value: Gemini ChatSession object
user_conversations = {}

# --- Telegram Bot Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    welcome_text = (
        "Hello! I'm **VetBot** 🐾, your AI assistant for pet health inquiries.\n\n"
        "You can send me:\n"
        "📝 **Text:** Ask me any question about your pet.\n"
        "🖼️ **Images:** Send a photo of a rash, injury, or anything you're concerned about.\n"
        "🎤 **Audio/Voice:** Send a recording of your pet's cough, wheeze, or other sounds.\n"
        "📹 **Videos:** Show me a clip of your pet's behavior, such as limping or seizures.\n\n"
        "I'll do my best to provide helpful information. For a list of commands, use /help. Let's get started!"
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message when the /help command is issued."""
    help_text = (
        "**Here's how I can help:**\n\n"
        "I can provide preliminary information and general advice about your pet's health. You can interact with me by sending:\n\n"
        "• **Text:** Ask any question about your pet's health, behavior, or care.\n"
        "• **Photos:** Send a picture of a concern, like a skin condition or injury.\n"
        "• **Audio/Voice:** Record your pet's cough, wheeze, or any unusual sound.\n"
        "• **Videos:** Show me a behavior you are worried about, like limping or a seizure.\n\n"
        "Simply send your message or media, and I will analyze it and respond.\n\n"
        "Use /start to see the full welcome message."
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all incoming messages: text, photo, video, and audio."""
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id

    # Show a "typing..." status to the user.
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Initialize a new chat session if it's the user's first message.
    if user_id not in user_conversations:
        try:
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                system_instruction=SYSTEM_INSTRUCTION
            )
            user_conversations[user_id] = model.start_chat(history=[])
            logger.info(f"New chat session started for user_id: {user_id}")
        except Exception as e:
            logger.error(f"Failed to create Gemini model for user {user_id}: {e}")
            await update.message.reply_text("Sorry, I'm having trouble connecting to my brain right now. Please try again later.")
            return

    chat_session = user_conversations[user_id]
    prompt_parts = []
    temp_file_path = None

    try:
        # 1. Process media files (photo, video, audio)
        media_file = (
            update.message.photo[-1] if update.message.photo else
            update.message.video or
            update.message.audio or
            update.message.voice
        )

        if media_file:
            processing_msg = await update.message.reply_text(f"🐾 Processing your file, please wait...")
            file_data = await context.bot.get_file(media_file.file_id)
            file_mime_type = None
            if update.message.photo:
                file_mime_type = 'image/jpeg'
                temp_file_path = f"temp_{media_file.file_id}.jpg"
            else:
                file_mime_type = media_file.mime_type
                temp_file_path = f"temp_{media_file.file_id}"

            await file_data.download_to_drive(temp_file_path)
            logger.info(f"Uploading file to Gemini: {temp_file_path} with mime_type: {file_mime_type}")
            uploaded_file = genai.upload_file(path=temp_file_path, mime_type=file_mime_type)
            
            while uploaded_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                raise ValueError("File processing failed on the backend.")

            prompt_parts.append(uploaded_file)
            await processing_msg.edit_text("File processed! Analyzing now...")

        # 2. Add text/caption to the prompt
        text_content = update.message.text or update.message.caption
        if text_content:
            prompt_parts.append(text_content)

        if not prompt_parts:
            await update.message.reply_text("Sorry, I can only process text, images, audio, and video.")
            return

        # 3. Send the complete prompt to Gemini
        response = chat_session.send_message(prompt_parts)

        # --- MODIFICATION START ---
        # 4. Send the response, with a fallback for bad markdown
        try:
            # First, try to send with Markdown
            await update.message.reply_text(response.text, parse_mode='Markdown')
        except BadRequest as e:
            # If it fails because of bad formatting, send it as plain text
            if "can't find end of the entity" in str(e):
                logger.warning(f"Bad Markdown from Gemini. Sending as plain text. Error: {e}")
                await update.message.reply_text(response.text)
            else:
                # If it's a different error, we still want to know about it
                logger.error(f"A BadRequest error occurred: {e}")
                await update.message.reply_text("Sorry, I had trouble formatting my response. Please try again.")
        # --- MODIFICATION END ---

    except Exception as e:
        logger.error(f"An error occurred while handling a message: {e}")
        await update.message.reply_text("Sorry, an unexpected error occurred. Please try again.")
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(f"Exception while handling an update: {context.error}")

def main() -> None:
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        logger.critical("FATAL ERROR: API keys (TELEGRAM_BOT_TOKEN, GEMINI_API_KEY) are not set.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- Register Handlers ---
    # Command handler for /start
    application.add_handler(CommandHandler("start", start_command))
    
    # Command handler for /help
    application.add_handler(CommandHandler("help", help_command))

    # Message handler for all supported media types and text (but not commands)
    all_messages_filter = (filters.TEXT & ~filters.COMMAND) | filters.PHOTO | filters.VIDEO | filters.AUDIO | filters.VOICE
    application.add_handler(MessageHandler(all_messages_filter, handle_message))

    # Error handler
    application.add_error_handler(error_handler)

    logger.info("VetBot is starting... Polling for updates.")
    application.run_polling()

if __name__ == "__main__":
    main()
