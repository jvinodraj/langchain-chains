from models import get_model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = get_model()

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()

# 1. Create a chain without the parser first
message_chain = prompt | model

# 2. Invoke it to get the full AIMessage object
ai_message = message_chain.invoke({'topic': 'Jesus'})

# 3. Print token usage metadata directly from the message
print("Token Usage:", getattr(ai_message, "usage_metadata", "Metadata not supported by this model provider"))

# 4. Parse the result into a plain string just like before
parser = StrOutputParser()
result = parser.invoke(ai_message)

print("\n--- Result ---")
print(result)

# (Optional) If you still want the graph visualization of the message chain
message_chain.get_graph().print_ascii()