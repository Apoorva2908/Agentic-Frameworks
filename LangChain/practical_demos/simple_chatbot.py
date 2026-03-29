#building simple chatbot without context first and then adding history to it
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

##Adding context or user history to prompt to provide context
chat_history = [
    SystemMessage(content= "You are a helpful assistant")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print("AI:", result.content)

print(chat_history)

## how to deal with large context conversations?