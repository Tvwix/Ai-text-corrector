import os
import torch
import torch_xla.core.xla_model as xm
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from fastapi import FastAPI
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.utils import executor
from dotenv import load_dotenv

# Загружаем токен бота из .env
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

# Инициализируем FastAPI
app = FastAPI()

# Загружаем модель mT5-large
model_name = "google/mt5-large"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(xm.xla_device())  # Загружаем на TPU

# Функция для исправления текста
def correct_text(text):
    input_text = "fix: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(xm.xla_device())

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)

    corrected_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return corrected_text

# Создаём Telegram-бота
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=["start"])
async def start(message: Message):
    await message.answer("Привет! Отправь мне текст, и я исправлю ошибки.")

@dp.message_handler()
async def fix_text(message: Message):
    corrected = correct_text(message.text)
    await message.reply(f"Исправленный текст:\n{corrected}")

# FastAPI endpoint для веб-приложения
class TextRequest:
    def __init__(self, text: str):
        self.text = text

@app.post("/fix_text")
async def fix_text(request: TextRequest):
    corrected = correct_text(request.text)
    return {"corrected_text": corrected}

# Запуск бота и FastAPI
if __name__ == "__main__":
    from aiogram import executor
    import uvicorn
    from threading import Thread

    # Запуск Telegram-бота в отдельном потоке
    def start_telegram():
        executor.start_polling(dp, skip_updates=True)

    thread = Thread(target=start_telegram)
    thread.start()

    # Запуск FastAPI сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)
  
