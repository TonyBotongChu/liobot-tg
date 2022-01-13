import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

import yaml

from app import BertBackend

with open("bot.yaml", 'r') as stream:
    settings = yaml.safe_load(stream)

token = settings['token']

REQUEST_KWARGS = {
    'proxy_url': settings['proxy']['url']
}

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

bot_backend = BertBackend("ernie-1.0")

# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def answer(update: Update, context: CallbackContext) -> None:
    template = ["您天天都在", (2, 5), "，您完全", (4, 7), "是吗？"]
    text = bot_backend.fill_template(template, update.message.text)
    text = bot_backend.fill_mask(text)
    update.message.reply_text(text)


def fill_template(update: Update, context: CallbackContext) -> None:
    template_name = context.args[0]
    text = context.args[1]
    if template_name == "健身环":
        template = ["您天天都在", (2, 5), "，您完全不", (3, 10), "是吗？"]
    else:
        template = ["error: unknown template ____liobot"]
    text = bot_backend.fill_template(template, text)
    text = bot_backend.fill_mask(text)
    update.message.reply_text(text)


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.

    updater = Updater(token, request_kwargs=REQUEST_KWARGS)
    # updater = Updater(token)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    dispatcher.add_handler(CommandHandler("say", fill_template))

    # on non command i.e message - echo the message on Telegram
    # dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, answer))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
