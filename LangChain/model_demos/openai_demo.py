# from langchain_openai import OpenAI --> for working directly with openai APIs
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv 
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
deployment = os.getenv("DEPLOYMENT")
print(api_key)

llm = AzureChatOpenAI(
    azure_deployment= deployment,  # IMPORTANT
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)

#Invoke is a method of langchain
response = llm.invoke("Summarize the deepseek paper in brief, highlight all the important points. \
 Also explain how is the architecture of deep seek different than that of GPT or equivalent")
print(response.content)

