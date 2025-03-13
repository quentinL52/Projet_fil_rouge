structure du projet :
/data :
- postgresml_offers
  contient les differents elements relatifs aux données tel que les connections et creations des tables dans postgresml ainsi que l'insertion
  des données dessus, la creation de table et insertion de données est obsoléte dans ce code car elle est geré dans MAGE.
- postgreml_reco
  contient les codes pour le chargement et l'extraction des données du CV.
  une partie du code est consacré a la créations de la table et le stockage du CV (le stockage n'est pas automatisé dans le code)

/devellopement :
partie de code en chantier 

/models : 
- agent_recommandation
  c'est l'agent qui va faire la recommandation a partir d'un CV stocké dans la base, dans un soucis de limite de resource materiel et d'api
  le code utilise a ce stade l'API openai.
  pour la partie agent les agents et les taches sont separé du reste pour plus de clarté dans le code et les agents utilise crewai
- config 
  gestion de l'url de connection de la base de données ainsi que les differents prompts utilisé par le modéle (simulateur)
- interview simulator 
  utilise langchain et llama 3 pour fonctionner, connection a la base de données des offres en selectionnant la premiére a ce stade 

/prompt
  contient le prompt pour le modéle 

/notebook
notebook de travail ainsi qu'un notebook des differentes block pour mage afin d'extraire les données 

/interface
  interface streamlit simpliste qui fait tourner le simulateur d'entretient

API nécéssaire pour faire tourner les modéles :
-GROQ 
-openai

données utilisé par les modéles :
l'extraction des données se fait par le biais d'un pipeline (disponible dans mes autres projets) qui utilise mage.ai pour scrapper des
des données et les stocker dans un bucket minio entre chaque etape de traitement.
les données sont ensuite stocké sur postgres, et les informations utiles sont ensuite transformé en embeddings qui sont stocké dans postgreml.
par defaut un jeux de données avec des embedding est disponible dans le repo