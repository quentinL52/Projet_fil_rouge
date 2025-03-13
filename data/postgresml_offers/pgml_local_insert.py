# Obsoléte, l'extraction et l'embedding des données est automatisé avec mage
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Lire le CSV
df = pd.read_csv(r"#path_to_csv") # changer le path 
print(f"CSV lu avec succès. {len(df)} lignes")
df = df.fillna('')

# Paramètres de connexion
params = {
    "host": "localhost",
    "port": "5433",
    "database": "projet_fil_rouge",
    "user": "postgres",
    "password": "postgres"
}

try:
    # Connexion à la base de données
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            print("Connexion à la base de données")

            # Vérifier les doublons avant l'insertion des données 
            print("Vérification des doublons")
            for index, row in df.iterrows():
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM data_fil_rouge 
                    WHERE entreprise = %s 
                    AND publication = %s 
                    AND poste = %s 
                    AND contrat = %s 
                    AND ville = %s 
                    AND lien = %s 
                    AND description_poste = %s;
                """, (row["entreprise"], row["publication"], row["poste"], row["contrat"], row["ville"], row["lien"], row["description_poste"]))
                if cur.fetchone()[0] > 0:
                    print(f"Doublon trouvé et supprimé : {row}")
                    df.drop(index, inplace=True)

            # Préparer les données
            print("Préparation des données")
            tuples = [tuple(x) for x in df.to_numpy()]
            cols = ','.join(list(df.columns))
            query = f"INSERT INTO data_fil_rouge ({cols}) VALUES %s"

            # Insérer les nouvelles données
            if len(tuples) > 0:
                print("Insertion des nouvelles données")
                execute_values(cur, query, tuples)

                # Générer les embeddings pour les nouvelles descriptions de poste
                print("Génération des embeddings")
                cur.execute("""
                    UPDATE data_fil_rouge 
                    SET embedding = pgml.embed('intfloat/e5-large', description_poste)
                    WHERE embedding IS NULL;
                """)

                conn.commit()
                print("Nouvelles données insérées!")
            else:
                print("Aucune nouvelle donnée à insérer.")

except psycopg2.Error as e:
    print(f"Erreur PostgreSQL: {e}")
except Exception as e:
    print(f"Erreur : {e}")