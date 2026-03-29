from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

#only few models support API inference , check inference available models
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.2",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

#----------------------------------------
#Demo without output parsers
#----------------------------------------
#1st prompt -> detailed report
template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

#2nd prompt -> Summary
template2 = PromptTemplate(
    template = 'Write a five line summary on the following text. /n {text}',
    input_variables = ['text']
)

prompt1 = template1.invoke({'topic':"black_hole"})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result1.content})

result = model.invoke(prompt2)
print(result.content)

#----------------------------------------
#Demo with output parsers
#----------------------------------------
print("with output parser")
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})
print(result)

##with output parsers flows becomes easier and clean
