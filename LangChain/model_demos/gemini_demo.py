from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
import os

load_dotenv()

#Gemini free tier is available for some models
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

#Invoke is a method of langchain
response = llm.invoke("Explain gradient descent simply")
print(response.text)


###Code to check the list of google models
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv 
# import os

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# for m in genai.list_models():
#     print(m.name)