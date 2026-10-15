import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)

llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
print(llm.invoke("Say hello in one short sentence.").content)