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
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    max_completion_tokens = 50
)

#Invoke is a method of langchain
response = llm.invoke("Explain gradient descent simply")
print(response.content)