# configuration des differents agents 
from crewai import Agent
from crewai.llm import LLM
from models.agent_recommandation.tool.tools import ExtractCVDataTool, GetSimilarJobOffersTool

openai_model = LLM(model="gpt-4")

def get_cv_analyzer(candidate_name, extract_cv_tool):
    return Agent(
        role="Analyste de CV",
        goal=f"Extraire les compétences clés, l'expérience et les qualifications du CV de {candidate_name}",
        backstory=f"Vous êtes un expert en analyse de CV chargé d'analyser UNIQUEMENT le CV authentique de {candidate_name} sans y ajouter d'information fictive.",
        llm=openai_model,
        tools=[extract_cv_tool],
        verbose=True
    )

def get_job_matcher(candidate_name, job_offers_tool):
    return Agent(
        role="Expert en correspondance d'emploi",
        goal=f"Analyser les offres d'emploi pour {candidate_name} et expliquer pourquoi elles correspondent à son profil",
        backstory=f"Vous êtes un expert en recrutement qui peut identifier précisément comment les compétences de {candidate_name} correspondent aux offres d'emploi.",
        llm=openai_model,
        tools=[job_offers_tool],
        verbose=True
    )

def get_recommendation_agent(candidate_name):
    return Agent(
        role="Conseiller en carrière",
        goal=f"Fournir des recommandations personnalisées à {candidate_name} basées sur son CV et les offres identifiées",
        backstory=f"Vous êtes un conseiller en carrière expérimenté qui aide {candidate_name} à optimiser son approche pour les offres qui lui correspondent le mieux.",
        llm=openai_model,
        verbose=True
    )