from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1 = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=500
)
model2 = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=500
)

prompt1 = PromptTemplate(
    template = 'generate short and simple notes from the following text \n{text}',
    input_variables = ['text']
)
prompt2 = PromptTemplate(
    template = 'Generate a 5 short question answers  from the following text \n{text}',
    input_variables = ['text']
)
prompt3 = PromptTemplate(
    template="""
Create a study document using the following content.

## Notes
{notes}

## Question Answers
{quiz}

Do not rewrite the story.
Only organize the notes and questions under the headings.
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Long ago, there was a tiny village hidden deep within a peaceful forest. The villagers lived happily, but every night the forest became so dark that no one dared to leave their homes.

At the heart of the forest stood an ancient oak tree. According to legend, a magical lantern was hidden inside its trunk. It was said that whoever found the lantern and lit it with kindness instead of fire would fill the entire forest with light.

One evening, a curious young girl named Lily decided to search for the lantern. She wasn't the strongest or the smartest person in the village, but she had a kind heart and never gave up. Along the way, she helped an injured rabbit, shared her food with a hungry traveler, and guided a lost child back to their family.

When Lily finally reached the ancient oak tree, the lantern appeared before her. As soon as she touched it, it began to glow without a flame. The light spread through the forest, illuminating every tree, river, and pathway. The darkness disappeared, and the villagers could finally walk freely at night.

The village celebrated Lily's courage, but she smiled and said, "The lantern didn't shine because of me. It shone because kindness lights the way."

From that day onward, the magical lantern reminded everyone that true strength comes from compassion, generosity, and helping others.

Moral: Kindness may seem small, but it has the power to brighten even the darkest places.
"""

result = chain.invoke({"text":text})
print(result)