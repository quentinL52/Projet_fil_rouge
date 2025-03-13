import json
import re
import pandas as pd
import warnings
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
groq = ChatGroq(api_key=groq_api, model_name="llama-3.3-70b-versatile")

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return pages

def parse_cv(pdf_path):
    pages = load_pdf(pdf_path)
    full_text = "\n".join([page.page_content for page in pages])

    prompt_template = """
    Vous êtes un expert en analyse de CV. Votre tâche est d'extraire les informations clés d'un CV et de les organiser sous forme de JSON STRICT.
    Le JSON doit contenir les clés suivantes :
    - "name": le nom complet du candidat (string)
    - "cv_content":  un résumé COMPACT et COMPLET du CV, incluant les compétences, l'expérience et la formation.  Ce champ doit contenir TOUTES les informations pertinentes du CV, dans un format texte lisible.

    Voici le texte du CV :
    {resume_text}

    Retournez UNIQUEMENT le JSON, sans texte introductif ni commentaires.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["resume_text"])

    chain = prompt | groq
    response = chain.invoke({"resume_text": full_text})
    response_str = response.content if hasattr(response, "content") else response

    match = re.search(r"\{.*\}", response_str, re.DOTALL)
    json_str = ""

    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            json_str = json.dumps(data)
        except json.JSONDecodeError:
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            json_str = json_str[start:end]
    else:
        print("Aucun JSON trouvé dans la réponse.")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Erreur lors du parsing du JSON: {e}")
        print(f"Texte JSON problématique : {json_str}")
        data = {}

    if data and "name" in data and "cv_content" in data:
        df = pd.DataFrame([{"name": data["name"], "cv_content": data["cv_content"]}])
    else:
        df = pd.DataFrame(columns=["name", "cv_content"])

    return df