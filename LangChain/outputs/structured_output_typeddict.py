from typing import TypedDict, Annotated, Optional
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("DEPLOYMENT")
print(api_key)


llm = AzureChatOpenAI(
    azure_deployment= deployment,  # IMPORTANT
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

#schema
class Review(TypedDict):
    summary: Annotated[str, "a brief summary of the review"]
    sentiment: Annotated[float, "a sentiment score between 0 and 1"]
    name: Annotated[Optional[str], "Extract the movie name ONLY if explicitly mentioned in the text. If not present, return null. Do NOT guess or infer."]

structured_model = llm.with_structured_output(Review)

result = structured_model.invoke("I loved the movie, the best part about the movie was the choice and placement of the music. Though it was too long, but it was worth the watch.")

print(result)