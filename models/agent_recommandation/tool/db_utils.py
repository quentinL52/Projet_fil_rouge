from sqlalchemy import text
from models.config import get_engine

def get_cv_data(name): # va recuperer le CV stocker dans la base de données 
    query = """
    SELECT name, cv_content
    FROM public.candidat
    WHERE name LIKE :name
    LIMIT 1;
    """
    try:
        with get_engine().connect() as conn:
            print(f"Exécution de la requête SQL pour récupérer le CV de: {name}")
            result = conn.execute(text(query), {"name": f"%{name}%"})
            cv_data = result.fetchone()
            
            if cv_data:
                print(f"CV trouvé pour {name}, longueur: {len(cv_data[1]) if cv_data[1] else 0} caractères")
                return cv_data[0], cv_data[1]
            
            print(f"Aucun CV trouvé pour {name}")
            return None, None
            
    except Exception as e:
        print(f"Erreur lors de la récupération du CV: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def find_similar_job_offers(candidate_name=None, limit=10): # module pour le matching par similarité d'embeddings
    query = """
    WITH candidate_embedding AS (
        SELECT embedding
        FROM public.candidat
        WHERE name LIKE :name
        LIMIT 1
    )
    SELECT
        df.entreprise,
        df.poste,
        LEFT(df.description_poste, 200) AS resume_poste,
        1 - (df.embedding <=> candidate_embedding.embedding) AS similarity
    FROM public.data_fil_rouge df, candidate_embedding
    ORDER BY similarity DESC
    LIMIT :limit;
    """
    params = {'name': f'%{candidate_name}%', 'limit': limit}
    
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text(query), params)
            return result.fetchall()
    except Exception as e:
        print(f"Erreur lors de la recherche d'offres: {e}")
        return []