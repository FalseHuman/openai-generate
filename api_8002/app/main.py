from fastapi import FastAPI, HTTPException
from yandex_cloud_ml_sdk import YCloudML
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

app = FastAPI()

# Инициализация SDK Yandex Cloud ML
sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),  # Ваш Folder ID
    auth=os.getenv("YANDEX_IAM_TOKEN")  # Ваш IAM-токен
)

# Выбор модели и настройка параметров
model = sdk.models.completions("llama", model_version="latest").configure(
    temperature=0.3,  # Параметр "творчества"
    max_tokens=500  # Максимальное количество токенов в ответе
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-text/")
async def generate_text_endpoint(request: PromptRequest):
    try:
        # Формирование запроса
        messages = [{"role": "user", "text": request.prompt}]

        # Выполнение запроса к Yandex Cloud ML
        result = model.run(messages)

        # Возвращаем результат
        if result:
            return {"result": result[0].text}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate text")

    except Exception as e:
        # Логируем ошибку
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Корневой эндпоинт
@app.get("/")
def read_root():
    return {"message": "Hello from API on port 8002"}