from fastapi import FastAPI, HTTPException
from openai import OpenAI, RateLimitError, APIError
import time
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

app = FastAPI()

# Инициализация клиента OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Функция для генерации текста с обработкой ошибок и задержками
def generate_text(prompt: str, max_retries: int = 3, delay: int = 5):
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            if completion.choices:
                return completion.choices[0].message.content
            else:
                return None
        except RateLimitError as e:
            wait_time = int(e.response.headers.get('Retry-After', 60))  # Получаем время ожидания из заголовка
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)  # Ждем указанное время
            continue
        except APIError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise HTTPException(status_code=500, detail="API Error")
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise HTTPException(status_code=500, detail="Internal Server Error")
    return None

# Эндпоинт для генерации текста
@app.post("/generate-text/")
async def generate_text_endpoint(prompt: str):
    result = generate_text(prompt)
    if result:
        return {"result": result}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate text")

# Корневой эндпоинт
@app.get("/")
def read_root():
    return {"message": "Hello from API on port 8000"}