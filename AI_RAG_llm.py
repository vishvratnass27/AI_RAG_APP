from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="ARTH.csv" , encoding="utf-8" , csv_args={
            'delimiter': '|',
            
        })

dataset = loader.load()

from dontenv import load_dotenv 

load_dotenv()

import os

openAIApi_key = os.getenv("llm_api_key")


from langchain.embeddings.openai import OpenAIEmbeddings

myembedmodel= OpenAIEmbeddings( openai_api_key=  openAIApi_key)


from langchain.vectorstores import FAISS

db = FAISS.from_documents(documents=dataset , embedding=myembedmodel)


def retrieve_data(query):
    similar_response = db.similarity_search(query)
    entire_data = [doc.page_content for doc in similar_response]
    return entire_data

from langchain.chat_models import ChatOpenAI

myllm = ChatOpenAI(model = "gpt-4" , openai_api_key=openAIApi_key , temperature = 0 )

ARTHtemplate = """ 
You are a world class business developnent representative for ARTH Program at LinuxWorId informatics pvt ltd.
I will share a prospect •s message with you and you will give me the best answer that
I should send to this prospect based on past best practies,
and you will follow ALL of the rules below:
1/ Response should be very similar or even Identical to the past best practies,
in terms of length, ton of voice, logical arguments and other details
2/ If the best practice are irrelevant, then try to minic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:

"""



from langchain.prompts import PromptTemplate

myprompt = PromptTemplate( template=ARTHtemplate , input_variables=["message" , "best_practice"])



from langchain.chains import LLMChain

mychain = LLMChain(llm=myllm , prompt=myprompt)

def ai_RAG_response(message):
    best_practice = retrieve_data(message)
    response = mychain.run(message=message ,best_practice=best_practice )
    
    return response


import streamlit as st


def main():
    st.set_page_config(
        page_title= "ARTH Customer response by AI" , page_icon=":bird:"")
    st.hearder("Customer response generator :bird:")
    message = st.text_area("my customer message")
    
    if message:
        st.write("Generative AI Arth message...")
        
        result = ai_RAG_response(message)
        
        st.info(result)
        
if __name__ == '__main__':
        main()
