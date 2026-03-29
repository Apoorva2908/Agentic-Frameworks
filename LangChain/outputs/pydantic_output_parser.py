from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

#only few models support API inference , check inference available models
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.2",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):

    name: str = Field(description = "Name of the person")
    age: int = Field(gt = 18, description = "age of the person")
    city: str = Field(description = "Name of the city the person belongs to")


parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = 'Give me the details of a {place} person. \n {format_instruction}',
    input_variables = ['place'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
#fills before runtime
)

prompt = template.invoke({'place':'Australian'})
print(prompt)

chain = template | model | parser
result = chain.invoke(prompt)
print(result)

