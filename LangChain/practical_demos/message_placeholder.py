from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

#chat_template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human','{query}')
])

chat_history = [HumanMessage('I want my refund for order #123')]

print(chat_history)
prompt = chat_template.invoke({'domain':'medical','chat_history':chat_history, 'query':'where is my refund'})
print(prompt)

