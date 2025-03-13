# configuration des taches des agents 
from crewai import Task

def create_analyze_cv_task(candidate_name, cv_content, cv_analyzer): # a travailler concernant l'extraction et l'analyse des données 
    return Task(
        description=f"Analyser le CV fourni de {candidate_name} et extraire les compétences clés, l'expérience et les qualifications.",
        expected_output="Une analyse détaillée et structurée des compétences, expériences et qualifications du candidat.",
        agent=cv_analyzer,
        input=f"""
Vous êtes un analyste de CV expert. Votre tâche est d'analyser le CV de {candidate_name} et d'en extraire les informations essentielles.

1. Utilisez l'outil extract_cv_data pour analyser le CV fourni. Cet outil vous renverra le contenu du CV.
2. À partir du contenu du CV, identifiez et organisez :
   - L'expérience professionnelle (postes, entreprises, dates)
   - Les compétences techniques et non techniques
   - Les formations et qualifications (diplômes, certifications)
   - Autres informations pertinentes (langues, projets significatifs, etc.)

3. Présentez votre analyse de manière structurée sous forme de liste pour faciliter la lecture.

IMPORTANT : Basez votre analyse UNIQUEMENT sur les informations présentes dans le CV. N'inventez pas d'informations.

Contenu du CV à analyser :
{cv_content}
"""
    )

def create_match_jobs_task(candidate_name, top_job_offers, job_matcher):
    return Task(
        description=f"Analysez les offres d'emploi fournies pour {candidate_name} et expliquez pourquoi elles correspondent à son profil.",
        expected_output="Analyse détaillée de la correspondance entre le profil du candidat et chacune des offres d'emploi.",
        agent=job_matcher,
        input=f"""
Votre mission est d'analyser en profondeur la correspondance entre le profil du candidat {candidate_name} et les offres d'emploi disponibles.

Voici l'analyse du CV de {candidate_name}:
{{analyze_cv_task.output}}

Pour obtenir les offres d'emploi qui correspondent le mieux au profil de {candidate_name}, utilisez l'outil get_similar_job_offers en lui fournissant le nom exact du candidat: "{candidate_name}".

Ensuite, pour chaque offre d'emploi retrouvée, analysez en détail:
1. Le degré de correspondance entre les compétences du candidat et celles requises pour le poste
2. L'adéquation de l'expérience professionnelle du candidat avec les exigences du poste
3. Comment les qualifications et formations du candidat correspondent aux prérequis du poste
4. Les points forts spécifiques du candidat qui seraient particulièrement valorisés pour ce poste

Si aucune offre n'est disponible, analysez quels types de postes seraient les plus adaptés au profil de {candidate_name} en fonction de son CV, et suggérez des pistes de recherche d'emploi.

Votre analyse doit être détaillée et précise, en vous basant uniquement sur les informations réelles du CV et des offres d'emploi.
"""
    )

def create_recommendation_task(candidate_name, recommendation_agent):
    return Task(
        description=f"Fournissez des recommandations personnalisées à {candidate_name} basées sur l'analyse de son CV et des correspondances d'emploi identifiées.",
        expected_output="Recommandations détaillées et personnalisées pour aider le candidat à optimiser ses candidatures.",
        agent=recommendation_agent,
        input=f"""
En tant que conseiller en carrière, votre mission est de fournir des recommandations personnalisées et concrètes à {candidate_name} qui l'aideront dans sa recherche d'emploi.

Voici l'analyse du CV de {candidate_name}:
{{analyze_cv_task.output}}

Voici l'analyse des correspondances d'emploi pour {candidate_name}:
{{match_jobs_task.output}}

Sur la base de ces informations, veuillez fournir des recommandations structurées qui couvrent les aspects suivants:

1. ADAPTATION DU CV ET DE LA LETTRE DE MOTIVATION
   - Comment {candidate_name} devrait adapter son CV pour mettre en valeur les compétences les plus pertinentes
   - Des suggestions spécifiques pour la lettre de motivation selon les types de postes visés

2. PRÉPARATION AUX ENTRETIENS
   - Les points clés à mettre en avant lors des entretiens
   - Comment présenter son expérience de manière convaincante
   - Des réponses possibles aux questions difficiles liées à son parcours

3. DÉVELOPPEMENT PROFESSIONNEL
   - Des compétences supplémentaires à acquérir pour améliorer son employabilité
   - Des certifications ou formations qui pourraient renforcer son profil

4. STRATÉGIE DE RECHERCHE D'EMPLOI
   - Les canaux de recherche les plus appropriés (plateformes, réseaux, approche directe)
   - Comment exploiter son réseau professionnel existant

Vos recommandations doivent être concrètes, personnalisées et directement applicables pour {candidate_name}.
"""
    )