import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

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

llm = ChatAnthropic(temperature=0.0,
                    model="claude-3-7-sonnet-20250219",
                    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"))

chain = summary_prompt | llm

res = chain.invoke(input={"information": "Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. He is best known for his mass-energy equivalence formula E=mc², which has been dubbed 'the world's most famous equation'. Einstein was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect."})
print(res.content)