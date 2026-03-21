from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv 
import os

load_dotenv()

#no free tier for claude
llm = ChatAnthropic(
    model = 'claude-sonnet-4-6'
)

#Invoke is a method of langchain
response = llm.invoke("Explain gradient descent simply")
print(response.content)