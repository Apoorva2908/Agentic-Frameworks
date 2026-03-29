from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
import os

load_dotenv()

#only few models support API inference , check inference available models
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.2",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

#creating response schema
schema = [
    ResponseSchema(name = 'fact_1', description = 'fact 1 about the topic'),
    ResponseSchema(name = 'fact_2', description = 'fact 2 about the topic'),
    ResponseSchema(name = 'fact_3', description = 'fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 facts about the {topic}. \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
#fills before runtime
)

prompt = template.format()
print(prompt)

chain = template | model | parser
result = chain.invoke({'topic': 'eifel tower'})
print(result)
