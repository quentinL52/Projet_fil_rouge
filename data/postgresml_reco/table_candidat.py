# obsolete sert uniquement à créer la table candidat dans la base de données postgresql
import psycopg2

params = {
    "host": "localhost",
    "port": "5433",
    "database": "projet_fil_rouge",
    "user": "postgres",
    "password": "postgres"
}

try:
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            create_table_query = """
            CREATE TABLE candidat (
                name TEXT,
                cv_content TEXT,
                embedding vector(1024)
            );
            """
            cur.execute(create_table_query)
            print("Connexion à la base de données")
            print("Activation de l'extension pgvector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("Connexion réussie !")
    conn.close()
except Exception as e:
    print(f"Erreur de connexion : {e}")



