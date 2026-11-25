from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template = 'generate a detailed report on {topic}',
    input_variables = ['topic']
)

propmt2 = PromptTemplate(
    template = 'Generate a 5 pointer summary from the folowing text \n{text}',
    input_variables = ['text']
)
model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()

chain = prompt1 | model | parser | propmt2 | model | parser

result = chain.invoke({"topic": "Moon"})
print(result)
chain.get_graph().print_ascii()