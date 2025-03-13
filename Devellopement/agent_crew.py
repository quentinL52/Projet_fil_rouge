# ok mais limite de tokens avec groq 
from langchain_groq import ChatGroq
import yaml
from crewai import Agent, Task, Crew
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sqlalchemy import create_engine, text
from config import get_engine
from typing import List, Dict
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential
from crewai.tools import tool
import litellm
def analyze_match_without_llm(candidate_data, job_offers):
    """Réalise une analyse préliminaire sans utiliser le LLM pour réduire la charge"""
    # Extraire les compétences du candidat
    cv_content = candidate_data.get('cv_content_original', candidate_data.get('cv_content', ''))
    skills = []
    technologies = []
    
    # Extraction simple des compétences et technologies
    if "Compétences" in cv_content:
        skills_section = cv_content.split("Compétences")[1].split("\n\n")[0]
        skills = [s.strip() for s in skills_section.split(',')]
    
    if "Outils & Technologies" in cv_content:
        tech_section = cv_content.split("Outils & Technologies")[1].split("\n\n")[0]
        technologies = [t.strip() for t in tech_section.split(',')]
    
    # Pour chaque offre, calculer un matching simple
    for job in job_offers:
        job_desc = job.get('description_poste', '')
        
        # Initialisation des forces et faiblesses
        job['matching_skills'] = []
        job['missing_skills'] = []
        
        # Analyse simple des correspondances de mots-clés
        for skill in skills + technologies:
            if skill and len(skill) > 3:  # Ignorer les compétences trop courtes
                if skill.lower() in job_desc.lower():
                    job['matching_skills'].append(skill)
        
        # Extraction de mots-clés des offres d'emploi
        job_keywords = []
        tech_keywords = ['python', 'sql', 'power bi', 'scikit-learn', 'pandas', 'r', 
                         'machine learning', 'data science', 'data viz', 'statistiques']
        
        for keyword in tech_keywords:
            if keyword in job_desc.lower() and keyword not in [t.lower() for t in technologies]:
                job['missing_skills'].append(keyword)
        
        # Préparation d'un résumé pour chaque offre
        job['basic_summary'] = {
            'entreprise': job.get('entreprise', ''),
            'poste': job.get('poste', ''),
            'match_score': job.get('similarity_score', 0),
            'points_forts': job.get('matching_skills', [])[:3],  # Limiter à 3 pour la concision
            'points_faibles': job.get('missing_skills', [])[:3]
        }
    
    return {
        'candidate_summary': {
            'name': candidate_data.get('name', ''),
            'skills': skills[:5],  # Limiter à 5 compétences
            'technologies': technologies[:5]  # Limiter à 5 technologies
        },
        'job_matches': job_offers
    }
import warnings
warnings.filterwarnings('ignore')


# Chargement des variables d'environnement
load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
engine = get_engine()

# Cache pour stocker les résultats intermédiaires
results_cache = {}

# Configuration du LLM avec paramètres encore plus réduits
llm = ChatGroq(model="groq/llama-3.1-8b-instant",  # Notez le préfixe "groq/"
               max_tokens=6000,  
               temperature=0.3,  
               groq_api_key=groq_api)

# Classes Pydantic pour la structure des données
class SkillGap(BaseModel):
    skill: str = Field(..., description="Compétence à développer")
    importance: str = Field(..., description="Niveau d'importance (Élevé, Moyen, Faible)")
    recommandation: str = Field(..., description="Recommandation pour développer cette compétence")

# Adaptation du modèle pour correspondre exactement aux colonnes disponibles
class JobMatch(BaseModel):
    entreprise: str = Field(default="Non spécifié", description="Nom de l'entreprise")
    publication: str = Field(default="", description="Date de publication")
    poste: str = Field(default="Non spécifié", description="Intitulé du poste")
    contrat: str = Field(default="Non spécifié", description="Type de contrat")
    ville: str = Field(default="Non spécifié", description="Localisation")
    lien: str = Field(default="", description="Lien vers l'offre")
    description_poste: str = Field(default="", description="Description du poste")
    similarity_score: float = Field(default=0.0, description="Score de similarité avec le CV")
    forces: List[str] = Field(default_factory=list, description="Points forts du candidat pour ce poste")
    faiblesses: List[str] = Field(default_factory=list, description="Points faibles du candidat pour ce poste")
    skills_a_developper: List[SkillGap] = Field(default_factory=list, description="Compétences à développer pour ce poste")

class FinalReport(BaseModel):
    candidate_name: str = Field(..., description="Nom du candidat")
    candidate_profile_summary: str = Field(..., description="Résumé du profil du candidat")
    top_job_matches: List[JobMatch] = Field(..., description="Top 5 des offres d'emploi correspondantes")
    recommandation_generale: str = Field(..., description="Recommandation générale pour améliorer l'employabilité")

# Décorateur de retry avec backoff exponentiel pour gérer les limites de débit
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=120))
def call_llm_with_retry(agent, prompt):
    try:
        # Ajouter un délai plus long pour respecter les limites de débit
        time.sleep(random.uniform(5.0, 10.0))
        return agent.llm.call(prompt)
    except Exception as e:
        print(f"Erreur LLM rencontrée: {e}")
        # Attendre plus longtemps après une erreur
        time.sleep(random.uniform(15.0, 30.0))
        raise

# Fonction pour extraire les informations essentielles du CV
def extract_cv_summary(cv_content, max_length=300):
    """Extrait un résumé concis du CV"""
    if len(cv_content) <= max_length:
        return cv_content
    
    lines = cv_content.split('\n')
    summary = []
    
    for line in lines:
        if "Résumé" in line or "Compétences" in line or "Outils & Technologies" in line:
            # Ajouter cette ligne et la suivante
            summary.append(line)
            if lines.index(line) + 1 < len(lines):
                summary.append(lines[lines.index(line) + 1])
    
    # Si on n'a pas assez d'informations, prendre le début du CV
    if len('\n'.join(summary)) < 100:
        return cv_content[:max_length] + "..."
    
    # Sinon retourner le résumé compilé
    result = '\n'.join(summary)
    if len(result) > max_length:
        return result[:max_length] + "..."
    return result

def get_candidate_data(candidate_name=None):
    """Version optimisée pour limiter la taille des données"""
    cache_key = f"candidate_{candidate_name}"
    if cache_key in results_cache:
        return results_cache[cache_key]
    
    # Requête avec les colonnes exactes de votre table
    query = """
    SELECT name, cv_content, embedding 
    FROM public.candidat 
    WHERE (:candidate_name IS NULL AND name = (SELECT name FROM public.candidat ORDER BY name LIMIT 1))
    OR name = :candidate_name
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"candidate_name": candidate_name})
        # Utiliser mappings() pour convertir le résultat en dictionnaire
        row = result.mappings().fetchone()
        if row is None:
            raise Exception(f"Aucun candidat trouvé avec le nom: {candidate_name}")
        
        candidate_data = dict(row)
        
        # Tronquer le contenu du CV pour réduire la taille des données
        if 'cv_content' in candidate_data and candidate_data['cv_content']:
            candidate_data['cv_content_original'] = candidate_data['cv_content']  # Sauvegarder l'original
            candidate_data['cv_content'] = extract_cv_summary(candidate_data['cv_content'])
        
        results_cache[cache_key] = candidate_data
        return candidate_data

# Fonction pour tronquer les descriptions des offres d'emploi
def truncate_job_description(description, max_length=300):
    """Tronque la description d'une offre d'emploi pour limiter la quantité de données"""
    if description and len(description) > max_length:
        return description[:max_length] + "..."
    return description

def get_job_offers_with_similarity(candidate_name=None, limit=5):  # Réduit à 5 offres maximum
    """Version optimisée pour limiter la taille des données"""
    cache_key = f"job_matches_{candidate_name}_{limit}"
    if cache_key in results_cache:
        return results_cache[cache_key]
    
    # Requête avec les colonnes exactes de votre table
    query = """
    WITH candidate_embedding AS (
        SELECT embedding
        FROM public.candidat
        WHERE (:candidate_name IS NULL AND name = (SELECT name FROM public.candidat ORDER BY name LIMIT 1))
        OR name = :candidate_name
    )
    SELECT 
        df.entreprise, 
        df.publication,
        df.poste, 
        df.contrat, 
        df.ville, 
        df.lien, 
        df.description_poste,
        1 - (df.embedding <=> candidate_embedding.embedding) AS similarity_score
    FROM public.data_fil_rouge df, candidate_embedding
    ORDER BY similarity_score DESC
    LIMIT :limit;
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query), {"candidate_name": candidate_name, "limit": limit})
        # Utiliser mappings() pour obtenir des dictionnaires
        job_matches = [dict(row) for row in result.mappings().fetchall()]
        
        # Tronquer les descriptions pour réduire la taille des données
        for job in job_matches:
            if 'description_poste' in job and job['description_poste']:
                job['description_poste'] = truncate_job_description(job['description_poste'])
        
        results_cache[cache_key] = job_matches
        return job_matches

# Outils pour les agents CrewAI
@tool
def fetch_candidate_data(candidate_name=None):
    """Récupère les données complètes du candidat depuis la base de données"""
    return get_candidate_data(candidate_name)

@tool
def fetch_job_offers_with_similarity(candidate_name=None, limit=10):
    """Récupère les offres d'emploi les plus pertinentes directement avec les scores de similarité"""
    return get_job_offers_with_similarity(candidate_name, limit)

# Agents optimisés - Réduits à 3 agents au lieu de 5
cv_job_analyzer = Agent(
    role="Analyste CV et Emploi",
    goal="Analyser le CV du candidat et identifier les emplois pertinents",
    backstory="Expert en analyse de CV et en recherche d'emploi qui peut identifier les compétences clés d'un candidat et les offres d'emploi correspondantes.",
    tools=[fetch_candidate_data, fetch_job_offers_with_similarity],
    llm=llm,
    verbose=True
)

strength_weakness_analyzer = Agent(
    role="Analyste de Forces et Faiblesses",
    goal="Analyser les points forts et les points faibles du candidat pour chaque offre d'emploi",
    backstory="Consultant en carrière spécialisé dans l'identification des forces et faiblesses d'un candidat par rapport à des offres spécifiques.",
    llm=llm,
    verbose=True
)

report_compiler = Agent(
    role="Compilateur de Rapport",
    goal="Compiler toutes les analyses en un rapport complet et actionnable",
    backstory="Rédacteur expert spécialisé dans la synthèse d'informations complexes et la formulation de recommandations claires.",
    llm=llm,
    verbose=True
)

# Tâches optimisées et divisées en plus petites unités
analyze_cv_task = Task(
    description="""
    Analyser brièvement le CV du candidat pour identifier:
    1. Ses 3-5 principales compétences techniques 
    2. Ses 3-5 principales compétences non-techniques
    3. Son profil professionnel en une phrase
    Soyez extrêmement concis - maximum 150 mots au total.
    """,
    agent=cv_job_analyzer,
    expected_output="Une liste très concise des compétences et du profil"
)

analyze_first_job_task = Task(
    description="""
    Pour la première offre d'emploi identifiée:
    1. Lister 2-3 points forts du candidat pour ce poste
    2. Lister 2-3 points faibles ou compétences manquantes
    3. Fournir une recommandation en une phrase
    Soyez extrêmement concis - maximum 100 mots au total.
    """,
    agent=strength_weakness_analyzer,
    expected_output="Analyse concise d'une seule offre d'emploi",
    context=[analyze_cv_task]
)

analyze_second_job_task = Task(
    description="""
    Pour la deuxième offre d'emploi identifiée:
    1. Lister 2-3 points forts du candidat pour ce poste
    2. Lister 2-3 points faibles ou compétences manquantes
    3. Fournir une recommandation en une phrase
    Soyez extrêmement concis - maximum 100 mots au total.
    """,
    agent=strength_weakness_analyzer,
    expected_output="Analyse concise d'une seule offre d'emploi",
    context=[analyze_cv_task]
)

compile_micro_report_task = Task(
    description="""
    Compiler un mini-rapport incluant:
    1. Un résumé du profil du candidat en 2-3 phrases
    2. Les 2 meilleures offres avec leurs points forts/faibles
    3. Une recommandation générale en une phrase
    Soyez extrêmement concis - maximum 200 mots au total.
    """,
    agent=report_compiler,
    expected_output="Un mini-rapport très concis",
    context=[analyze_cv_task, analyze_first_job_task, analyze_second_job_task],
    output_pydantic=FinalReport
)

# Création du crew optimisé
job_recommendation_crew = Crew(
    agents=[cv_job_analyzer, strength_weakness_analyzer, report_compiler],
    tasks=[analyze_cv_task, analyze_first_job_task, analyze_second_job_task, compile_micro_report_task],
    verbose=True
)

# Fonction pour exécuter le crew avec gestion des erreurs et meilleure gestion du rate limit
def run_job_recommendation(candidate_name=None):
    try:
        # Vider le cache avant chaque exécution
        results_cache.clear()
        
        # Étape 1: Récupérer les données du candidat (avec résumé tronqué)
        print("Récupération préalable des données du candidat...")
        try:
            candidate_data = get_candidate_data(candidate_name)
            results_cache["candidate_data"] = candidate_data
            print(f"Candidat trouvé: {candidate_data['name']}")
        except Exception as db_error:
            print(f"Erreur lors de la récupération des données du candidat: {db_error}")
            return {"error": f"Erreur base de données: {str(db_error)}"}
        
        # Étape 2: Récupérer les offres d'emploi avec leurs scores de similarité (seulement 3)
        print("Récupération préalable des offres d'emploi pertinentes...")
        try:
            job_matches = get_job_offers_with_similarity(candidate_name, limit=3)  # Limité à 3 pour réduire la charge
            results_cache["job_matches"] = job_matches
            print(f"Nombre d'offres trouvées: {len(job_matches)}")
        except Exception as job_error:
            print(f"Erreur lors de la récupération des offres d'emploi: {job_error}")
            return {"error": f"Erreur lors de la récupération des offres: {str(job_error)}"}
        
        # Étape 3: Réaliser une analyse préliminaire sans LLM
        print("Réalisation d'une analyse préliminaire...")
        try:
            preliminary_analysis = analyze_match_without_llm(candidate_data, job_matches)
            results_cache["preliminary_analysis"] = preliminary_analysis
        except Exception as analysis_error:
            print(f"Erreur lors de l'analyse préliminaire: {analysis_error}")
            # Continuer même en cas d'erreur dans l'analyse préliminaire
        
        # Étape 4: Exécuter le crew avec les données préchargées et un délai entre les tâches
        print("Démarrage de l'analyse avec les agents...")
        time.sleep(10)  # Attendre 10 secondes avant de commencer les appels LLM
        
        # Ajouter les données importantes au contexte global du crew
        if hasattr(job_recommendation_crew, 'set_shared_memory'):
            job_recommendation_crew.set_shared_memory({
                'candidate': candidate_data,
                'jobs': job_matches[:2]  # Seulement les 2 premiers emplois
            })
        
        result = job_recommendation_crew.kickoff(
            inputs={
                "candidate_name": candidate_name,
                "job_count": 2,  # Indication qu'on veut analyser seulement 2 emplois
                "max_words": 50  # Indication du nombre maximum de mots par réponse
            }
        )
        return result
    except Exception as e:
        print(f"Erreur lors de l'exécution du crew: {e}")
        # Tenter de générer un rapport simplifié avec les données que nous avons
        if "candidate_data" in results_cache and "job_matches" in results_cache:
            print("Utilisation des résultats partiels pour générer un rapport simplifié...")
            try:
                # Si on a fait l'analyse préliminaire, utiliser ces résultats
                if "preliminary_analysis" in results_cache:
                    analysis = results_cache["preliminary_analysis"]
                    candidate_summary = analysis.get('candidate_summary', {})
                    job_matches = analysis.get('job_matches', [])[:3]
                else:
                    candidate_data = results_cache["candidate_data"]
                    job_matches = results_cache["job_matches"][:3]
                    candidate_summary = {
                        'name': candidate_data.get('name', 'Inconnu'),
                        'skills': [],
                        'technologies': []
                    }
                
                # Rapport de base construit à partir des données prétraitées
                basic_report = {
                    "candidate_name": candidate_summary.get('name', 'Inconnu'),
                    "candidate_profile_summary": f"Candidat avec des compétences en {', '.join(candidate_summary.get('skills', [])[:3])} et maîtrisant {', '.join(candidate_summary.get('technologies', [])[:3])}",
                    "top_job_matches": []
                }
                
                # Ajouter les offres d'emploi
                for job in job_matches:
                    job_entry = {
                        "entreprise": job.get("entreprise", ""),
                        "poste": job.get("poste", ""),
                        "contrat": job.get("contrat", ""),
                        "ville": job.get("ville", ""),
                        "lien": job.get("lien", ""),
                        "similarity_score": job.get("similarity_score", 0),
                        "forces": job.get("matching_skills", [])[:3] or ["Non analysé - erreur LLM"],
                        "faiblesses": job.get("missing_skills", [])[:3] or ["Non analysé - erreur LLM"],
                        "skills_a_developper": [
                            {"skill": skill, "importance": "Moyen", "recommandation": f"Développer la compétence en {skill}"}
                            for skill in job.get("missing_skills", [])[:2]
                        ] or [{"skill": "Non analysé", "importance": "Non analysé", "recommandation": "Non analysé"}]
                    }
                    basic_report["top_job_matches"].append(job_entry)
                
                # Ajouter une recommandation générale
                basic_report["recommandation_generale"] = "Approfondir les compétences techniques mentionnées dans les offres et mettre en avant l'expérience en analyse de données et utilisation des outils de visualisation."
                
                return {"partial_report": basic_report, "error": str(e)}
            except Exception as report_error:
                print(f"Erreur lors de la génération du rapport simplifié: {report_error}")
                # En cas d'échec, revenir au rapport ultra basique
                candidate_data = results_cache["candidate_data"]
                job_matches = results_cache["job_matches"][:3]
                
                basic_report = {
                    "candidate_name": candidate_data.get("name", "Inconnu"),
                    "candidate_profile_summary": "Données extraites de la base sans analyse LLM",
                    "top_job_matches": [
                        {
                            "entreprise": job.get("entreprise", ""),
                            "poste": job.get("poste", ""),
                            "contrat": job.get("contrat", ""),
                            "ville": job.get("ville", ""),
                            "lien": job.get("lien", ""),
                            "similarity_score": job.get("similarity_score", 0),
                            "forces": ["Non analysé - erreur LLM"],
                            "faiblesses": ["Non analysé - erreur LLM"],
                            "skills_a_developper": [
                                {"skill": "Non analysé", "importance": "Non analysé", "recommandation": "Non analysé"}
                            ]
                        } for job in job_matches
                    ],
                    "recommandation_generale": "Rapport simplifié généré suite à une erreur de limite de débit (rate limit)"
                }
                return {"partial_report": basic_report, "error": str(e)}
        return {"error": str(e)}

if __name__ == "__main__":
    result = run_job_recommendation()
    
    # Afficher le résultat sous forme de dictionnaire ou JSON formaté
    if isinstance(result, dict) and "error" in result:
        print("ERREUR RENCONTRÉE:")
        print(result["error"])
        if "partial_report" in result:
            print("\nRAPPORT PARTIEL GÉNÉRÉ:")
            import json
            print(json.dumps(result["partial_report"], indent=2, ensure_ascii=False))
    else:
        print(result)