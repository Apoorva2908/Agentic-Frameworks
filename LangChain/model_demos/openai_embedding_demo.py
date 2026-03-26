from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

model_name = "text-embedding-3-small"
deployment = "text-embedding-3-small"

api_version = "2024-02-01"

embedding = AzureOpenAIEmbeddings(
    azure_deployment= deployment,  
    api_version= api_version
)

answer = embedding.embed_query("Delhi is the capital of India")

print(str(answer))
#embed_documents in place of embed_query for multiple queries
ques = embedding.embed_query("What is the capital of India")
print(str(ques))

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_similarity(answer, ques))