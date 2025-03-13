import psycopg2
import pandas as pd

def insert_into_db(df):
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
                if not df.empty:
                    for _, row in df.iterrows():
                        cur.execute(
                            "INSERT INTO candidat (name, cv_content) VALUES (%s, %s)",
                            (row['name'], row['cv_content'])
                        )
                    conn.commit()
                    cur.execute("""
                        UPDATE candidat
                        SET embedding = pgml.embed('intfloat/e5-large', cv_content)
                        WHERE embedding IS NULL;
                    """)
                    conn.commit()
                    print("Données insérées avec succès.")
                else:
                    print("Aucune nouvelle donnée à insérer.")

    except psycopg2.Error as e:
        print(f"Erreur PostgreSQL: {e}")
    except Exception as e:
        print(f"Erreur : {e}")