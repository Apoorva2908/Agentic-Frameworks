from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
deployment = os.getenv("DEPLOYMENT")

model = AzureChatOpenAI(
    azure_deployment= deployment,  # IMPORTANT
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT")

)

messages = [
    SystemMessage(content = "You are a helpful female assistant"),
    HumanMessage(content = "tell me about LangChain in 50 words")
]

result = model.invoke(messages)

messages.append(AIMessage(result.content))
print(messages)
