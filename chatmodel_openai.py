from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

model = ChatOpenAI(
    model="openrouter/free", temperature=1.5
)

result = model.invoke("Write the story of Mahatma Gandhi in 100 words.")
print(result.content)