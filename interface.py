# page streamlit basique pour le simulateur d'entretient
import streamlit as st
import warnings
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import create_engine, text
from models.config import read_system_prompt, chatbot_instructions_centered

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()

st.set_page_config(page_title="Chatbot Interface", layout="wide")
st.title("Chatbot Interface")

groq_api = os.getenv('GROQ_API_KEY')
groq = ChatGroq(api_key=groq_api, model_name="llama-3.3-70b-versatile")

file_path = r'prompts\rag_prompt.txt'
instruction = r'prompts\instructions.txt'
prompt_rh = read_system_prompt(file_path)
instructions_simu = chatbot_instructions_centered(instruction)

db_url = "postgresql+psycopg2://postgres:postgres@localhost:5433/projet_fil_rouge"
engine = create_engine(db_url)

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_rh),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | groq

def analyze_with_postgresml(text_input):
    with engine.connect() as connection:
        query = text("""
            SELECT pgml.transform(
                task => 'text-classification',
                inputs => ARRAY[:text_input]
            ) AS analysis;
        """)
        result = connection.execute(query, {"text_input": text_input}).fetchone()
    return result[0] if result else None

with engine.connect() as connection:
    query = text("SELECT entreprise, poste, description_poste FROM data_fil_rouge")
    result = connection.execute(query).fetchone()

if result:
    entreprise, poste, description = result
    
    st.write(instructions_simu)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        initial_response = chain.invoke({
            "input": "Bonjour",
            "entreprise": entreprise,
            "poste": poste,
            "description": description,
            "chat_history": []
        })
        st.session_state.messages.append({"role": "assistant", "content": initial_response.content})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Vous :"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        analysis = analyze_with_postgresml(prompt)
        if analysis:
            st.write(f"Analyse PostgresML : {analysis}")

        with st.chat_message("assistant"):
            response = chain.invoke({
                "input": prompt,
                "entreprise": entreprise,
                "poste": poste,
                "description": description,
                "chat_history": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            })
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})

else:
    st.error("Aucune donnée trouvée dans la base de données.")
