import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Streaming Bot",page_icon ='🤖')

st.title("Streaming Bot")

# get response
def get_response(query,chat_history):
    template =  """
    You are a helpful assistant and Expert Coder. Answer the following question considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(groq_api_key=groq_api_key,
               model_name = "Llama3-8b-8192")
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "chat_history":chat_history,
        "user_question":query
    })
    



# conversation
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)



# user input
user_query = st.chat_input("Your message")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        ai_response= get_response(user_query,st.session_state.chat_history)
        st.markdown(ai_response)
    
    st.session_state.chat_history.append(AIMessage(ai_response))


              