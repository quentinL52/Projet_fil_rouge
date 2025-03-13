from crewai.tools import BaseTool
from langchain.tools import tool

class ExtractCVDataTool(BaseTool):
    name: str = "extract_cv_data"
    description: str = "Extrait les informations pertinentes d'un CV pour analyser les compétences, l'expérience et les qualifications"

    def _run(self, cv_content: str) -> str:
        print(f"ExtractCVDataTool REÇOIT: {len(cv_content)} caractères")
    def _run(self, cv_content: str) -> str:
        if not cv_content or not isinstance(cv_content, str) or len(cv_content.strip()) < 20:
            return "CV invalide ou trop court. Veuillez fournir un contenu de CV complet."

        return f"""
Analyse du CV effectuée. Longueur du CV: {len(cv_content)} caractères.

Contenu complet du CV pour analyse:
---
{cv_content}
---

Veuillez extraire les informations suivantes de ce CV:
1. Expérience professionnelle
2. Compétences techniques
3. Compétences non techniques
4. Formation et diplômes
5. Certifications et qualifications
6. Projets significatifs (si présents)
7. Langues (si présentes)
8. Autres informations pertinentes
"""

class GetSimilarJobOffersTool(BaseTool):
    name: str = "get_similar_job_offers"
    description: str = "Récupère les 3 offres d'emploi les plus similaires au profil du candidat en utilisant la similarité d'embeddings"

    def _run(self, candidate_name: str) -> str:
        from models.agent_recommandation.tool.db_utils import find_similar_job_offers
        
        print(f"GetSimilarJobOffersTool: Recherche d'offres pour {candidate_name}")        
        job_offers = find_similar_job_offers(candidate_name, limit=3)
        
        if not job_offers:
            return f"Aucune offre d'emploi trouvée pour {candidate_name}."        
        formatted_offers = f"Voici les 3 offres d'emploi les plus pertinentes pour {candidate_name} (classées par ordre de similarité décroissante):\n\n"
        
        for i, offer in enumerate(job_offers, 1):
            entreprise, poste, description, score = offer
            formatted_offers += f"OFFRE #{i} - Score de similarité: {score:.2f}\n"
            formatted_offers += f"Poste: {poste}\n"
            formatted_offers += f"Entreprise: {entreprise}\n"
            formatted_offers += f"Description: {description}\n"
            formatted_offers += "-" * 50 + "\n\n"
        
        return formatted_offers