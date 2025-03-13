import os
from dotenv import load_dotenv
from crewai import Crew, Process, Task
from models.agent_recommandation.tool.db_utils import get_cv_data, find_similar_job_offers
from models.agent_recommandation.tool.tools import ExtractCVDataTool, GetSimilarJobOffersTool
from models.agent_recommandation.config.agent_config import (
    get_cv_analyzer,
    get_job_matcher,
    get_recommendation_agent
)
from models.agent_recommandation.config.task_config import (
    create_analyze_cv_task,
    create_match_jobs_task,
    create_recommendation_task
)

load_dotenv()

class JobRecommendationCrew:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("La clé API OpenAI n'est pas définie.")
        
        self.extract_cv_tool = ExtractCVDataTool()
        self.job_offers_tool = GetSimilarJobOffersTool()
    
    def kickoff(self, inputs):
        candidate_name = inputs.get('candidate_name')
        if not candidate_name:
            raise ValueError("Le nom du candidat est requis.")
        
        print(f"Démarrage de l'analyse pour le candidat: {candidate_name}")
        
        try:
            result = get_cv_data(candidate_name)
            if isinstance(result, tuple) and len(result) >= 2:
                name = result[0]
                cv_content = result[1]
            else:
                print(f"Format de retour inattendu pour get_cv_data: {result}")
                name = candidate_name
                cv_content = "CV non disponible - format de données incorrect."
            
            print(f"Données du candidat récupérées: Nom={name}, Longueur CV={len(cv_content) if cv_content else 0}")
            if cv_content and len(cv_content) > 100:
                print(f"Début du CV: {cv_content[:100]}...")
            job_offers_data = find_similar_job_offers(candidate_name, limit=3)
            top_job_offers = ""
            
            if job_offers_data and len(job_offers_data) > 0:
                print(f"Offres trouvées: {len(job_offers_data)}")
                for i, job in enumerate(job_offers_data, 1):
                    entreprise, poste, description, score = job
                    top_job_offers += f"\nOFFRE #{i} - Score: {score:.2f}\n"
                    top_job_offers += f"Poste: {poste}\n"
                    top_job_offers += f"Entreprise: {entreprise}\n"
                    top_job_offers += f"Description: {description}\n"
                    top_job_offers += "-" * 40 + "\n"
                    print(f"Offre {i}: {poste} chez {entreprise} (score: {score:.2f})")
            else:
                top_job_offers = "Aucune offre d'emploi similaire trouvée pour ce candidat."
                print("Aucune offre trouvée")
                
            print(f"Nombre d'offres pré-récupérées: {len(job_offers_data) if job_offers_data else 0}")
            cv_analyzer = get_cv_analyzer(name, self.extract_cv_tool)
            job_matcher = get_job_matcher(name, self.job_offers_tool)
            recommendation_agent = get_recommendation_agent(name)
            
            analyze_cv_task = Task(
            description=f"Analyser le CV fourni de {name} et extraire les compétences clés, l'expérience et les qualifications.",
            expected_output="Une analyse détaillée et structurée des compétences, expériences et qualifications du candidat.",
            agent=cv_analyzer,
            input=f"""
        Vous êtes un analyste de CV expert. Votre tâche est d'analyser le CV de {name} et d'en extraire les informations essentielles.
        IMPORTANT: Analysez UNIQUEMENT le CV ci-dessous. N'inventez aucune information.

        CV COMPLET À ANALYSER:
        {cv_content}
        """
            )
            
            match_jobs_task = create_match_jobs_task(name, top_job_offers, job_matcher)
            recommendation_task = create_recommendation_task(name, recommendation_agent)
            crew = Crew(
                agents=[cv_analyzer, job_matcher, recommendation_agent],
                tasks=[analyze_cv_task, match_jobs_task, recommendation_task],
                process=Process.sequential,
                verbose=True
            )
            print("Démarrage de l'exécution du crew...")
            result = crew.kickoff()
            return result
            
        except Exception as e:
            print(f"Erreur lors de l'exécution du crew: {e}")
            import traceback
            traceback.print_exc()
            return f"Une erreur s'est produite: {str(e)}"