from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

#only few models support API inference , check inference available models
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.2",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("Suggest one business that i can start today with low effort and quick money")

print(result.content)