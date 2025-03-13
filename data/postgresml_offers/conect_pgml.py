import psycopg2
# permet de se connecter a la base de données, creer une table et activer l'extension pgvector.

# parametres de connection 
params = {
    "host": "localhost",
    "port": "5433",
    "database": "projet_fil_rouge",
    "user": "postgres",
    "password": "postgres"
}

try:
    # connection a la base de données 
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            # créer ma table 
            create_table_query = """
            CREATE TABLE data_fil_rouge (
                entreprise TEXT,
                publication TEXT,
                poste TEXT,
                contrat TEXT,
                ville TEXT,
                lien TEXT,
                description_poste TEXT,
                embedding vector(1024)
            );
            """
            cur.execute(create_table_query)
            print("Connexion à la base de données")
            # Activer l'extension pgvector
            print("Activation de l'extension pgvector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("Connexion réussie !")
    conn.close()
except Exception as e:
    print(f"Erreur de connexion : {e}")



