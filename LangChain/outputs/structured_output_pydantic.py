from pydantic import BaseModel, EmailStr, Field
from typing import Optional
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
class Student(BaseModel):
    name: str = "Apoorva" #can set optional values as well
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt = 0, lt = 10, description = "gives cgpa of a student")

new_student = {"name": 32}
student_2 = {"age":30, 'email':'abc@abc.com', 'cgpa':9} #doesn't work if email doesn't have @
# student = Student(**new_student)
student = Student(**student_2)

class Review(BaseModel):
    Summary: str = Field(description= "A brief summary of the review")#can set optional values as well
    sentiment: float = Field(gt = 0, lt = 1, description = "gives the review sentiment")
    name: Optional[str] = Field(default = None, description= "gives the name of the movie only if explicitly mentioned in the review")


print(student)
print(dict(student))

structured_model = llm.with_structured_output(Review)

result = structured_model.invoke("I loved the movie, the best part about the movie was the choice and placement of the music. Though it was too long, but it was worth the watch.")

print(result)