
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from rich import print as rprint
load_dotenv(override=True)
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")

model =init_chat_model(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)
#2、声明一个函数（工具）
def get_weather(city:str):
    return f"{city}天气晴朗~~~"
#3、将函数绑定在模型上
model_with_tools=model.bind_tools([get_weather])
#4、调用模型
response = model_with_tools.invoke("北京的天气怎么样")
rprint(response)