
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_langchain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-002",
        google_api_key="AIzaSyCwAMyKR7kGf3aTcTcX6BPIamPAbAMjb5Q",
        temperature=0.7,
        max_output_tokens=200
    )
    
    explain_prompt = PromptTemplate(
        input_variables=["prediction", "features"],
        template="Given a predicted grade of {prediction} based on features {features}, provide an brief explanation of the student's performance in a concise and friendly manner and give them tips to improve their performance."
    )
    
    chat_prompt = PromptTemplate(
        input_variables=["query"],
        template="You are a helpful assistant for students. Answer the following query in a brief and friendly manner: {query}"
    )
    
    explain_chain = LLMChain(llm=llm, prompt=explain_prompt)
    chat_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return {"explain": explain_chain, "chat": chat_chain}

def get_explanation(chains, prediction, features):
    return chains["explain"].run(prediction=prediction, features=features)

def get_chat_response(chains, query,context):
    
    query = "keeping this context in mind: " + context + "\n" + query
    return chains["chat"].run(query=query)