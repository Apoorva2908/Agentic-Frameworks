from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

#only few models support API inference , check inference available models
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.2",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me the name, age and city of a fictional person. \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
#fills before runtime
)

prompt = template.format()
print(prompt)


result = model.invoke(prompt)
print(result)

final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))


