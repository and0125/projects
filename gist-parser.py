"""
Simple example of using Ollama 3.2 with LangChain
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Load environment variables from .env file
load_dotenv()


if __name__ == "__main__":

    summary_template = """
    given the information {information} about a person, I want you to create:

    - a short summary of the person
    - two interesting facts about the person

"""

summary_prompt = PromptTemplate(
    input_variables=["information"],
    template=summary_template,
)

llm = ChatOllama(model="llama3.2")

chain = summary_prompt | llm | StrOutputParser()

res = chain.invoke(input={"information": "Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. He is best known for his mass-energy equivalence formula E=mc², which has been dubbed 'the world's most famous equation'. Einstein was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect."})

# print result: now its just a string response, not the full JSON object
print(res)