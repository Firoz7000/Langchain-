from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template= "generate 5 interesting facts about {topic}",
    input_variables= ["topic"]
)

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Stars"})
print(result)

chain.get_graph().print_ascii()