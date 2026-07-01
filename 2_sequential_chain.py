from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 
os.environ["LANGCHAIN_PROJECT"] = "Sequential App"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatMistralAI(
    model="ministral-3b-latest",
    temperature=0.7
)
model2 = ChatMistralAI(
    model="ministral-3b-latest",
    temperature=0.7
)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'tags':['llm', 'report generation', 'summarization'],
    'metadata': {'model1':'mistral','model2' : 'mistral'}
}

result = chain.invoke({'topic': 'Unemployment in India'},config = config)

print(result)
