import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from sqlalchemy import create_engine, text  

from config import read_system_prompt, chatbot_instructions_centered, get_engine
import shutil
load_dotenv()
# groq
groq_api = os.getenv('GROQ_API_KEY')
groq = ChatGroq(api_key=groq_api, model_name="llama-3.3-70b-versatile")
# prompt
file_path = r'prompts\rag_prompt.txt'
instruction = r'prompts\instructions.txt'
prompt_rh = read_system_prompt(file_path)
instructions_simu = chatbot_instructions_centered(instruction)
# Connexion à PostgresML
engine = get_engine()

prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_rh),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")              ])

memory = ChatMessageHistory(
        return_messages=True,
        output_key="output",
        input_key="input"  )

chain = prompt | groq

# PostgresML (retourne une analyse de sentiment avec un score mais reste a travailler sur le score et a l'affiner)
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

# Récupération des données de la premiére offre
with engine.connect() as connection:
    query = text("SELECT entreprise, poste, description_poste FROM data_fil_rouge")
    result = connection.execute(query).fetchone()

if result:
    entreprise, poste, description = result
    chat_history = []

    print(instructions_simu)

    # initialiser le modéle avec un premier bonjour pour qu'il commence a parler au lancement
    initial_response = chain.invoke({
        "input": "Bonjour",  
        "entreprise": entreprise,
        "poste": poste,
        "description": description,
        "chat_history": []
    })

    print("\nAssistant :", initial_response.content)
    chat_history.append(AIMessage(content=initial_response.content)) 

    while True:
        user_input = input("\nVous : ")
        if user_input.lower() == 'Au revoir':
            break

        # Analyse de sentiment avec PostgresML
        analysis = analyze_with_postgresml(user_input)
        if analysis:
            print(f"\nAnalyse PostgresML : {analysis}")

        # Génération de la réponse de l'assistant
        response = chain.invoke({
            "input": user_input,
            "entreprise": entreprise,
            "poste": poste,
            "description": description,
            "chat_history": chat_history
        })

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))

        print("\nAssistant :", response.content)