from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import os



load_dotenv()



print(os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))



llm = HuggingFaceEndpoint(

    repo_id="meta-llama/Llama-3.1-8B-Instruct",

    task="text-generation",

    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),

)



model = ChatHuggingFace(llm=llm)



result = model.invoke("What is the capital of France?")

print(result.content)